// Permutohedral lattice bilateral filter with runtime-dispatched pd/vd.
//
// Entrypoint:
//   void permuto_filter(float* weight_out, float* out,
//                       const float* values,   // [H, W, vd]
//                       const float* features, // [H, W, pd]
//                       void* matrix_storage,
//                       float* h_values_dev, float* blur_values_dev,
//                       int* h_entries_dev, signed short* h_keys_dev,
//                       int* blur_neighbors_dev,
//                       int pd, int vd, int w, int h);
//
// Out must be preallocated to [H, W, vd]. weight_out to [H, W, 1].
// Out is overwritten with the filtered output. weight_out receives
// the homogeneous normalizer (used by backward pass, unused here but
// kept in the signature to match the torch wrapper).
//
// This is the full Gaussian/bilateral permutohedral filter: splat, blur,
// then slice. The blur pass performs one [1, 2, 1] / 4 convolution along
// each of the PD+1 lattice directions.
//
// Based on Adams/Baek/Davis 2010 ("Fast high-dimensional filtering using the
// permutohedral lattice"), Stanford CUDA reference implementation.

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define BLOCK_SIZE 64   // 8x8 threads per block
#define MAX_SUPPORTED_PD 16
#define MAX_SUPPORTED_VD 8


// ---------- hash table ----------

__device__ __constant__ float*        table_values;
__device__ __constant__ signed short* table_keys;
__device__ __constant__ int*          table_entries;
__device__ __constant__ unsigned int  table_capacity;

// constants for fast mod-by-constant (Granlund-Montgomery)
__device__ __constant__ unsigned int __div_m;
__device__ __constant__ unsigned int __div_l;
__device__ __constant__ unsigned int __div_c;

// Full-blur scale factors for runtime-dispatched PD and blurVariance=0.5:
// (PD + 1) * sqrt(((1/6) + 0.5) / ((i + 1) * (i + 2))).
__device__ __constant__ float scaleFactor_const[MAX_SUPPORTED_PD];

// Hash offsets for lattice blur directions. Since permuto_hash is linear
// modulo 2^32, hash(key +/- direction[color]) can be formed from hash(key)
// +/- blurHashOffset_const[color].
__device__ __constant__ unsigned int blurHashOffset_const[MAX_SUPPORTED_PD + 1];

__device__ inline unsigned int modHash(unsigned int n) {
    unsigned int t1 = __umulhi(__div_m, n);
    return n - ((t1 + ((n - t1) >> 1)) >> (__div_l - 1)) * __div_c;
}

template<int PD>
__device__ __host__ static inline unsigned int permuto_hash(const signed short* key) {
    unsigned int k = 0;
    #pragma unroll
    for (int i = 0; i < PD; i++) {
        k += key[i];
        k = k * 2531011u;
    }
    return k;
}

template<int PD>
__device__ static int hashTableInsert(const signed short* key, unsigned int slot) {
    unsigned int fh = permuto_hash<PD>(key);
    int h = modHash(fh);
    while (1) {
        int* e = &table_entries[h];
        int contents = atomicCAS(e, -1, -2);
        if (contents == -2) {
            // locked by someone else, try next slot
        } else if (contents == -1) {
            // we locked an empty cell - write key
            #pragma unroll
            for (int i = 0; i < PD; i++) {
                table_keys[slot * PD + i] = key[i];
            }
            atomicExch(e, slot);
            return h;
        } else {
            // cell has a key - check match
            bool match = true;
            #pragma unroll
            for (int i = 0; i < PD && match; i++) {
                match = (table_keys[contents * PD + i] == key[i]);
            }
            if (match) return h;
        }
        h++;
        if (h == (int)(table_capacity * 2)) h = 0;
    }
}

template<int PD>
__device__ static int hashTableRetrieve(const signed short* key) {
    int h = modHash(permuto_hash<PD>(key));
    while (1) {
        int* e = table_entries + h;
        if (*e == -1) return -1;
        bool match = true;
        #pragma unroll
        for (int i = 0; i < PD && match; i++) {
            match = (table_keys[(*e) * PD + i] == key[i]);
        }
        if (match) return *e;
        h++;
        if (h == (int)(table_capacity * 2)) h = 0;
    }
}

template<int PD>
__device__ static int hashTableRetrieveWithHash(unsigned int hash, const signed short* key) {
    int h = modHash(hash);
    while (1) {
        int* e = table_entries + h;
        if (*e == -1) return -1;
        bool match = true;
        #pragma unroll
        for (int i = 0; i < PD && match; i++) {
            match = (table_keys[(*e) * PD + i] == key[i]);
        }
        if (match) return *e;
        h++;
        if (h == (int)(table_capacity * 2)) h = 0;
    }
}


// ---------- lattice stages ----------

struct MatrixEntry {
    int   index;
    float weight;
};

template<int PD, int VD>
__global__ static void createMatrix(const int w, const int h,
                                    const float* positions,
                                    MatrixEntry* matrix)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    const int idx = y * w + x;
    const bool outOfBounds = (x >= w) || (y >= h);

    float myElevated[PD + 1];
    int   myGreedy[PD + 1];
    int   myRank[PD + 1];
    float myBarycentric[PD + 2];
    __shared__ short keys[PD * BLOCK_SIZE];
    short* myKey = keys + threadId * PD;

    if (!outOfBounds) {
        const float* myPosition = positions + idx * PD;

        // Elevate pd-dim feature to (pd+1)-dim on H_d
        myElevated[PD] = -PD * myPosition[PD - 1] * scaleFactor_const[PD - 1];
        #pragma unroll
        for (int i = PD - 1; i > 0; i--) {
            myElevated[i] = (myElevated[i + 1]
                             - i * myPosition[i - 1] * scaleFactor_const[i - 1]
                             + (i + 2) * myPosition[i] * scaleFactor_const[i]);
        }
        myElevated[0] = myElevated[1] + 2 * myPosition[0] * scaleFactor_const[0];

        // Find the zero-colored lattice point greedily
        signed short sum = 0;
        #pragma unroll
        for (int i = 0; i <= PD; i++) {
            float v = myElevated[i] * (1.0f / (PD + 1));
            float up   = ceilf(v)  * (PD + 1);
            float down = floorf(v) * (PD + 1);
            myGreedy[i] = (up - myElevated[i] < myElevated[i] - down) ? (signed short)up
                                                                      : (signed short)down;
            sum += myGreedy[i];
        }
        sum /= PD + 1;

        // Permutation rank for the simplex
        #pragma unroll
        for (int i = 0; i <= PD; i++) {
            myRank[i] = 0;
            #pragma unroll
            for (int j = 0; j <= PD; j++) {
                float di = myElevated[i] - myGreedy[i];
                float dj = myElevated[j] - myGreedy[j];
                if (di < dj || (di == dj && i > j)) myRank[i]++;
            }
        }

        // Fix up rank if sum drifted
        if (sum > 0) {
            #pragma unroll
            for (int i = 0; i <= PD; i++) {
                if (myRank[i] >= PD + 1 - sum) {
                    myGreedy[i] -= PD + 1;
                    myRank[i]   += sum - (PD + 1);
                } else {
                    myRank[i]   += sum;
                }
            }
        } else if (sum < 0) {
            #pragma unroll
            for (int i = 0; i <= PD; i++) {
                if (myRank[i] < -sum) {
                    myGreedy[i] += PD + 1;
                    myRank[i]   += (PD + 1) + sum;
                } else {
                    myRank[i]   += sum;
                }
            }
        }

        // Barycentric coordinates in the canonical simplex
        #pragma unroll
        for (int i = 0; i <= PD + 1; i++) myBarycentric[i] = 0.0f;

        #pragma unroll
        for (int i = 0; i <= PD; i++) {
            float delta = (myElevated[i] - myGreedy[i]) * (1.0f / (PD + 1));
            myBarycentric[PD     - myRank[i]] += delta;
            myBarycentric[PD + 1 - myRank[i]] -= delta;
        }
        myBarycentric[0] += 1.0f + myBarycentric[PD + 1];
    }

    // Insert PD+1 lattice neighbors into the hash table
    #pragma unroll
    for (int color = 0; color <= PD; color++) {
        if (!outOfBounds) {
            #pragma unroll
            for (int i = 0; i < PD; i++) {
                myKey[i] = myGreedy[i] + color;
                if (myRank[i] > PD - color) myKey[i] -= (PD + 1);
            }
            MatrixEntry r;
            r.index  = hashTableInsert<PD>(myKey, idx * (PD + 1) + color);
            r.weight = myBarycentric[color];
            matrix[idx * (PD + 1) + color] = r;
        }
    }
}

template<int PD>
__global__ static void cleanHashTable(int n, MatrixEntry* matrix) {
    const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;
    if (idx >= n) return;

    int* e = table_entries + idx;
    if (*e >= 0) {
        // Rehash so races between dup-key inserters converge.
        *e = hashTableRetrieve<PD>(table_keys + (*e) * PD);
    }
}

template<int PD, int VD>
__global__ static void splatCache(const int w, const int h,
                                  const float* values,
                                  MatrixEntry* matrix)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + (blockIdx.y / (PD + 1)) * blockDim.y;
    const int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    const int color = blockIdx.y % (PD + 1);
    const int idx = y * w + x;
    const bool outOfBounds = (x >= w) || (y >= h);

    __shared__ int   sharedOffsets[BLOCK_SIZE];
    __shared__ float sharedValues[BLOCK_SIZE * (VD + 1)];

    int   myOffset = -1;
    float* myValue = sharedValues + threadId * (VD + 1);

    if (!outOfBounds) {
        const float* value = values + idx * VD;

        MatrixEntry r = matrix[idx * (PD + 1) + color];
        // convert hash-table entry index into keys/values-array index
        matrix[idx * (PD + 1) + color].index = r.index = table_entries[r.index];
        myOffset = sharedOffsets[threadId] = r.index * (VD + 1);

        #pragma unroll
        for (int j = 0; j < VD; j++) myValue[j] = value[j] * r.weight;
        myValue[VD] = r.weight;
    } else {
        sharedOffsets[threadId] = -1;
    }

    __syncthreads();

    if (outOfBounds) return;

    // Intra-block dedup: threads with the same offset merge into the lowest id.
    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (i < threadId) {
            if (myOffset == sharedOffsets[i]) return;   // someone earlier owns this key
        } else if (i > threadId) {
            if (myOffset == sharedOffsets[i]) {
                #pragma unroll
                for (int j = 0; j <= VD; j++) {
                    sharedValues[threadId * (VD + 1) + j] += sharedValues[i * (VD + 1) + j];
                }
            }
        }
    }

    float* val = table_values + myOffset;
    #pragma unroll
    for (int j = 0; j <= VD; j++) atomicAdd(val + j, myValue[j]);
}

template<int PD, int VD>
__global__ static void blur(const int n_vertices,
                            float* new_values,
                            MatrixEntry* matrix,
                            int* blur_neighbors,
                            const int color)
{
    const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;
    if (idx >= n_vertices) return;

    // Only canonical lattice vertices own storage in table_values.
    if (matrix[idx].index != idx) return;

    const int neighborBase = (idx * (PD + 1) + color) * 2;
    const int offNp = blur_neighbors[neighborBase];
    const int offNm = blur_neighbors[neighborBase + 1];

    const float* valMe = table_values + (VD + 1) * idx;
    float* valOut = new_values + (VD + 1) * idx;

    #pragma unroll
    for (int i = 0; i <= VD; i++) {
        float accum = 2.0f * valMe[i];
        if (offNp >= 0) accum += table_values[(VD + 1) * offNp + i];
        if (offNm >= 0) accum += table_values[(VD + 1) * offNm + i];
        valOut[i] = 0.25f * accum;
    }
}

template<int PD>
__global__ static void precomputeBlurNeighbors(const int n_vertices,
                                               MatrixEntry* matrix,
                                               int* blur_neighbors)
{
    const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;
    if (idx >= n_vertices) return;
    if (matrix[idx].index != idx) return;

    signed short myKey[PD];
    #pragma unroll
    for (int i = 0; i < PD; i++) myKey[i] = table_keys[idx * PD + i];

    const unsigned int myHash = permuto_hash<PD>(myKey);
    for (int color = 0; color <= PD; color++) {
        signed short np[PD];
        signed short nm[PD];
        #pragma unroll
        for (int i = 0; i < PD; i++) {
            np[i] = myKey[i] + 1;
            nm[i] = myKey[i] - 1;
        }
        if (color < PD) {
            np[color] -= PD + 1;
            nm[color] += PD + 1;
        }
        const int neighborBase = (idx * (PD + 1) + color) * 2;
        blur_neighbors[neighborBase] =
            hashTableRetrieveWithHash<PD>(myHash + blurHashOffset_const[color], np);
        blur_neighbors[neighborBase + 1] =
            hashTableRetrieveWithHash<PD>(myHash - blurHashOffset_const[color], nm);
    }
}

template<int PD, int VD>
__global__ static void slice(const int w, const int h,
                             float* values,
                             MatrixEntry* matrix,
                             float* weight_out)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int idx = y * w + x;
    if (x >= w || y >= h) return;

    float myValue[VD];

    #pragma unroll
    for (int i = 0; i < VD; i++) myValue[i] = 0.0f;
    float myWeight = 0.0f;

    #pragma unroll
    for (int i = 0; i <= PD; i++) {
        MatrixEntry r = matrix[idx * (PD + 1) + i];
        const float* val = table_values + r.index * (VD + 1);
        #pragma unroll
        for (int j = 0; j < VD; j++) myValue[j] += r.weight * val[j];
        myWeight += r.weight * val[VD];
    }

    myWeight = 1.0f / myWeight;
    #pragma unroll
    for (int j = 0; j < VD; j++) values[idx * VD + j] = myValue[j] * myWeight;
    weight_out[idx] = myWeight;
}

template<int PD, int VD>
__global__ static void g_splatCache(const int w, const int h,
                                    const float* values,
                                    MatrixEntry* matrix,
                                    const float* weight_in)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + (blockIdx.y / (PD + 1)) * blockDim.y;
    const int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    const int color = blockIdx.y % (PD + 1);
    const int idx = y * w + x;
    const bool outOfBounds = (x >= w) || (y >= h);

    __shared__ int   sharedOffsets[BLOCK_SIZE];
    __shared__ float sharedValues[BLOCK_SIZE * (VD + 1)];

    int myOffset = -1;
    float* myValue = sharedValues + threadId * (VD + 1);

    if (!outOfBounds) {
        const float* value = values + idx * VD;
        const float thisWeight = weight_in[idx];

        MatrixEntry r = matrix[idx * (PD + 1) + color];
        matrix[idx * (PD + 1) + color].index = r.index = table_entries[r.index];
        myOffset = sharedOffsets[threadId] = r.index * (VD + 1);

        #pragma unroll
        for (int j = 0; j < VD; j++) myValue[j] = value[j] * thisWeight * r.weight;
        myValue[VD] = r.weight;
    } else {
        sharedOffsets[threadId] = -1;
    }

    __syncthreads();

    if (outOfBounds) return;

    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (i < threadId) {
            if (myOffset == sharedOffsets[i]) return;
        } else if (i > threadId) {
            if (myOffset == sharedOffsets[i]) {
                #pragma unroll
                for (int j = 0; j <= VD; j++) {
                    sharedValues[threadId * (VD + 1) + j] += sharedValues[i * (VD + 1) + j];
                }
            }
        }
    }

    float* val = table_values + myOffset;
    #pragma unroll
    for (int j = 0; j <= VD; j++) atomicAdd(val + j, myValue[j]);
}

template<int PD, int VD>
__global__ static void g_slice(const int w, const int h,
                               float* values,
                               MatrixEntry* matrix)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int idx = y * w + x;
    if (x >= w || y >= h) return;

    float myValue[VD];

    #pragma unroll
    for (int i = 0; i < VD; i++) myValue[i] = 0.0f;

    #pragma unroll
    for (int i = 0; i <= PD; i++) {
        MatrixEntry r = matrix[idx * (PD + 1) + i];
        const float* val = table_values + r.index * (VD + 1);
        #pragma unroll
        for (int j = 0; j < VD; j++) myValue[j] += r.weight * val[j];
    }

    #pragma unroll
    for (int j = 0; j < VD; j++) values[idx * VD + j] = myValue[j];
}


// ---------- host entrypoint ----------

template<int PD>
static void setup_lattice_constants(int capacity)
{
    float scaleFactor_host[PD];
    unsigned int blurHashOffset_host[PD + 1];
    for (int i = 0; i < PD; i++) {
        scaleFactor_host[i] = (PD + 1) * sqrtf(((1.0f / 6.0f) + 0.5f) / ((i + 1) * (i + 2)));

        signed short offset[PD];
        for (int j = 0; j < PD; j++) offset[j] = 1;
        offset[i] -= PD + 1;
        blurHashOffset_host[i] = permuto_hash<PD>(offset);
    }
    signed short lastOffset[PD];
    for (int j = 0; j < PD; j++) lastOffset[j] = 1;
    blurHashOffset_host[PD] = permuto_hash<PD>(lastOffset);

    cudaMemcpyToSymbol(scaleFactor_const, scaleFactor_host, PD * sizeof(float));
    cudaMemcpyToSymbol(blurHashOffset_const, blurHashOffset_host, (PD + 1) * sizeof(unsigned int));

    unsigned long long two32 = ((unsigned long long)1) << 32;
    unsigned int div_c = 2u * capacity;
    unsigned int div_l = (unsigned int)ceilf(logf((float)div_c) / logf(2.0f));
    unsigned int div_m = (unsigned int)((two32 << div_l) / div_c - two32 + 1);
    cudaMemcpyToSymbol(__div_c, &div_c, sizeof(unsigned int));
    cudaMemcpyToSymbol(__div_l, &div_l, sizeof(unsigned int));
    cudaMemcpyToSymbol(__div_m, &div_m, sizeof(unsigned int));
}

template<int PD, int VD>
static void permuto_filter_impl(float* weight_out, float* out,
                                const float* values,
                                const float* features,
                                void* matrix_storage,
                                float* h_values_dev,
                                float* blur_values_dev,
                                int* h_entries_dev,
                                signed short* h_keys_dev,
                                int* blur_neighbors_dev,
                                int w, int h)
{
    const int n = w * h;

    MatrixEntry* matrix_dev = reinterpret_cast<MatrixEntry*>(matrix_storage);

    // Hash table: 2n(pd+1) entries, load factor 0.5.
    const int capacity = n * (PD + 1);
    cudaMemset(h_values_dev, 0, capacity * (VD + 1) * sizeof(float));
    cudaMemset(h_entries_dev, -1, capacity * 2 * sizeof(int));
    cudaMemset(h_keys_dev, 0, capacity * PD * sizeof(signed short));

    unsigned int cap_u = (unsigned int)capacity;
    cudaMemcpyToSymbol(table_capacity, &cap_u, sizeof(unsigned int));
    cudaMemcpyToSymbol(table_values,   &h_values_dev,  sizeof(float*));
    cudaMemcpyToSymbol(table_entries,  &h_entries_dev, sizeof(int*));
    cudaMemcpyToSymbol(table_keys,     &h_keys_dev,    sizeof(signed short*));

    setup_lattice_constants<PD>(capacity);

    dim3 blocks((w - 1) / 8 + 1, (h - 1) / 8 + 1, 1);
    dim3 blockSize(8, 8, 1);

    // 1. Build the matrix
    createMatrix<PD, VD><<<blocks, blockSize>>>(w, h, features, matrix_dev);

    // 2. Clean duplicate hash entries
    int cleanBlockSize = 32;
    dim3 cleanBlocks((n - 1) / cleanBlockSize + 1, 2 * (PD + 1), 1);
    cleanHashTable<PD><<<cleanBlocks, cleanBlockSize>>>(2 * n * (PD + 1), matrix_dev);

    // 3. Splat: accumulate values onto lattice vertices
    dim3 splatBlocks = blocks;
    splatBlocks.y *= (PD + 1);
    splatCache<PD, VD><<<splatBlocks, blockSize>>>(w, h, values, matrix_dev);

    // 4. Resolve blur neighbor indices once; blur passes then avoid hash probes.
    const int n_vertices = n * (PD + 1);
    precomputeBlurNeighbors<PD><<<cleanBlocks, cleanBlockSize>>>(
        n_vertices, matrix_dev, blur_neighbors_dev);

    // 5. Blur: convolve lattice values along each permutohedral direction.
    float* current_values_dev = h_values_dev;
    float* next_values_dev = blur_values_dev;
    for (int color = 0; color <= PD; color++) {
        blur<PD, VD><<<cleanBlocks, cleanBlockSize>>>(
            n_vertices, next_values_dev, matrix_dev, blur_neighbors_dev, color);
        cudaMemcpyToSymbol(table_values, &next_values_dev, sizeof(float*));
        float* previous_values_dev = current_values_dev;
        current_values_dev = next_values_dev;
        next_values_dev = previous_values_dev;
    }

    // 6. Slice: interpolate filtered values back to pixel grid.
    slice<PD, VD><<<blocks, blockSize>>>(w, h, out, matrix_dev, weight_out);
}

template<int PD, int VD>
static void permuto_filter_grad_impl(float* out,
                                     const float* grad_values,
                                     const float* weight_in,
                                     const float* features,
                                     void* matrix_storage,
                                     float* h_values_dev,
                                     float* blur_values_dev,
                                     int* h_entries_dev,
                                     signed short* h_keys_dev,
                                     int* blur_neighbors_dev,
                                     int w, int h)
{
    const int n = w * h;

    MatrixEntry* matrix_dev = reinterpret_cast<MatrixEntry*>(matrix_storage);

    const int capacity = n * (PD + 1);
    cudaMemset(h_values_dev, 0, capacity * (VD + 1) * sizeof(float));
    cudaMemset(h_entries_dev, -1, capacity * 2 * sizeof(int));
    cudaMemset(h_keys_dev, 0, capacity * PD * sizeof(signed short));

    unsigned int cap_u = (unsigned int)capacity;
    cudaMemcpyToSymbol(table_capacity, &cap_u, sizeof(unsigned int));
    cudaMemcpyToSymbol(table_values,   &h_values_dev,  sizeof(float*));
    cudaMemcpyToSymbol(table_entries,  &h_entries_dev, sizeof(int*));
    cudaMemcpyToSymbol(table_keys,     &h_keys_dev,    sizeof(signed short*));

    setup_lattice_constants<PD>(capacity);

    dim3 blocks((w - 1) / 8 + 1, (h - 1) / 8 + 1, 1);
    dim3 blockSize(8, 8, 1);

    createMatrix<PD, VD><<<blocks, blockSize>>>(w, h, features, matrix_dev);

    int cleanBlockSize = 32;
    dim3 cleanBlocks((n - 1) / cleanBlockSize + 1, 2 * (PD + 1), 1);
    cleanHashTable<PD><<<cleanBlocks, cleanBlockSize>>>(2 * n * (PD + 1), matrix_dev);

    dim3 splatBlocks = blocks;
    splatBlocks.y *= (PD + 1);
    g_splatCache<PD, VD><<<splatBlocks, blockSize>>>(w, h, grad_values, matrix_dev, weight_in);

    const int n_vertices = n * (PD + 1);
    precomputeBlurNeighbors<PD><<<cleanBlocks, cleanBlockSize>>>(
        n_vertices, matrix_dev, blur_neighbors_dev);

    float* current_values_dev = h_values_dev;
    float* next_values_dev = blur_values_dev;
    for (int color = PD; color >= 0; color--) {
        blur<PD, VD><<<cleanBlocks, cleanBlockSize>>>(
            n_vertices, next_values_dev, matrix_dev, blur_neighbors_dev, color);
        cudaMemcpyToSymbol(table_values, &next_values_dev, sizeof(float*));
        float* previous_values_dev = current_values_dev;
        current_values_dev = next_values_dev;
        next_values_dev = previous_values_dev;
    }

    g_slice<PD, VD><<<blocks, blockSize>>>(w, h, out, matrix_dev);
}

#define DISPATCH_VD(PD_VALUE) \
    switch (vd) { \
        case 1: permuto_filter_impl<PD_VALUE, 1>(weight_out, out, values, features, matrix_storage, h_values_dev, blur_values_dev, h_entries_dev, h_keys_dev, blur_neighbors_dev, w, h); return; \
        case 2: permuto_filter_impl<PD_VALUE, 2>(weight_out, out, values, features, matrix_storage, h_values_dev, blur_values_dev, h_entries_dev, h_keys_dev, blur_neighbors_dev, w, h); return; \
        case 3: permuto_filter_impl<PD_VALUE, 3>(weight_out, out, values, features, matrix_storage, h_values_dev, blur_values_dev, h_entries_dev, h_keys_dev, blur_neighbors_dev, w, h); return; \
        case 4: permuto_filter_impl<PD_VALUE, 4>(weight_out, out, values, features, matrix_storage, h_values_dev, blur_values_dev, h_entries_dev, h_keys_dev, blur_neighbors_dev, w, h); return; \
        case 5: permuto_filter_impl<PD_VALUE, 5>(weight_out, out, values, features, matrix_storage, h_values_dev, blur_values_dev, h_entries_dev, h_keys_dev, blur_neighbors_dev, w, h); return; \
        case 6: permuto_filter_impl<PD_VALUE, 6>(weight_out, out, values, features, matrix_storage, h_values_dev, blur_values_dev, h_entries_dev, h_keys_dev, blur_neighbors_dev, w, h); return; \
        case 7: permuto_filter_impl<PD_VALUE, 7>(weight_out, out, values, features, matrix_storage, h_values_dev, blur_values_dev, h_entries_dev, h_keys_dev, blur_neighbors_dev, w, h); return; \
        case 8: permuto_filter_impl<PD_VALUE, 8>(weight_out, out, values, features, matrix_storage, h_values_dev, blur_values_dev, h_entries_dev, h_keys_dev, blur_neighbors_dev, w, h); return; \
        default: return; \
    }

#define DISPATCH_VD_GRAD(PD_VALUE) \
    switch (vd) { \
        case 1: permuto_filter_grad_impl<PD_VALUE, 1>(out, grad_values, weight_in, features, matrix_storage, h_values_dev, blur_values_dev, h_entries_dev, h_keys_dev, blur_neighbors_dev, w, h); return; \
        case 2: permuto_filter_grad_impl<PD_VALUE, 2>(out, grad_values, weight_in, features, matrix_storage, h_values_dev, blur_values_dev, h_entries_dev, h_keys_dev, blur_neighbors_dev, w, h); return; \
        case 3: permuto_filter_grad_impl<PD_VALUE, 3>(out, grad_values, weight_in, features, matrix_storage, h_values_dev, blur_values_dev, h_entries_dev, h_keys_dev, blur_neighbors_dev, w, h); return; \
        case 4: permuto_filter_grad_impl<PD_VALUE, 4>(out, grad_values, weight_in, features, matrix_storage, h_values_dev, blur_values_dev, h_entries_dev, h_keys_dev, blur_neighbors_dev, w, h); return; \
        case 5: permuto_filter_grad_impl<PD_VALUE, 5>(out, grad_values, weight_in, features, matrix_storage, h_values_dev, blur_values_dev, h_entries_dev, h_keys_dev, blur_neighbors_dev, w, h); return; \
        case 6: permuto_filter_grad_impl<PD_VALUE, 6>(out, grad_values, weight_in, features, matrix_storage, h_values_dev, blur_values_dev, h_entries_dev, h_keys_dev, blur_neighbors_dev, w, h); return; \
        case 7: permuto_filter_grad_impl<PD_VALUE, 7>(out, grad_values, weight_in, features, matrix_storage, h_values_dev, blur_values_dev, h_entries_dev, h_keys_dev, blur_neighbors_dev, w, h); return; \
        case 8: permuto_filter_grad_impl<PD_VALUE, 8>(out, grad_values, weight_in, features, matrix_storage, h_values_dev, blur_values_dev, h_entries_dev, h_keys_dev, blur_neighbors_dev, w, h); return; \
        default: return; \
    }

void permuto_filter(float* weight_out, float* out,
                    const float* values,
                    const float* features,
                    void* matrix_storage,
                    float* h_values_dev,
                    float* blur_values_dev,
                    int* h_entries_dev,
                    signed short* h_keys_dev,
                    int* blur_neighbors_dev,
                    int pd, int vd, int w, int h)
{
    switch (pd) {
        case 1: DISPATCH_VD(1)
        case 2: DISPATCH_VD(2)
        case 3: DISPATCH_VD(3)
        case 4: DISPATCH_VD(4)
        case 5: DISPATCH_VD(5)
        case 6: DISPATCH_VD(6)
        case 7: DISPATCH_VD(7)
        case 8: DISPATCH_VD(8)
        case 9: DISPATCH_VD(9)
        case 10: DISPATCH_VD(10)
        case 11: DISPATCH_VD(11)
        case 12: DISPATCH_VD(12)
        case 13: DISPATCH_VD(13)
        case 14: DISPATCH_VD(14)
        case 15: DISPATCH_VD(15)
        case 16: DISPATCH_VD(16)
        default: return;
    }
}

void permuto_filter_grad(float* out,
                         const float* grad_values,
                         const float* weight_in,
                         const float* features,
                         void* matrix_storage,
                         float* h_values_dev,
                         float* blur_values_dev,
                         int* h_entries_dev,
                         signed short* h_keys_dev,
                         int* blur_neighbors_dev,
                         int pd, int vd, int w, int h)
{
    switch (pd) {
        case 1: DISPATCH_VD_GRAD(1)
        case 2: DISPATCH_VD_GRAD(2)
        case 3: DISPATCH_VD_GRAD(3)
        case 4: DISPATCH_VD_GRAD(4)
        case 5: DISPATCH_VD_GRAD(5)
        case 6: DISPATCH_VD_GRAD(6)
        case 7: DISPATCH_VD_GRAD(7)
        case 8: DISPATCH_VD_GRAD(8)
        case 9: DISPATCH_VD_GRAD(9)
        case 10: DISPATCH_VD_GRAD(10)
        case 11: DISPATCH_VD_GRAD(11)
        case 12: DISPATCH_VD_GRAD(12)
        case 13: DISPATCH_VD_GRAD(13)
        case 14: DISPATCH_VD_GRAD(14)
        case 15: DISPATCH_VD_GRAD(15)
        case 16: DISPATCH_VD_GRAD(16)
        default: return;
    }
}
