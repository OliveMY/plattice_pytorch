#ifndef MIRRORED_ARRAY_H
#define MIRRORED_ARRAY_H

template<typename T>
class MirroredArray {
  public:
    MirroredArray(size_t len) {
	size = len;
	host = new T[len];
	owner = true;	
#ifdef CUDA_MEMORY_H
	allocateCudaMemory((void**)&(device), len*sizeof(T));
#else
	cudaMalloc((void**)&(device), len*sizeof(T));
#endif
    }

    MirroredArray(T *data, size_t len) {
	size = len;
	host = data;
	owner = false;
	cudaMalloc((void**)&(device), len*sizeof(T));
	hostToDevice();
    }

    void hostToDevice() {
	cudaMemcpy(device, host, size*sizeof(T), cudaMemcpyHostToDevice);
    }
    
    void deviceToHost() {
	cudaMemcpy(host, device, size*sizeof(T), cudaMemcpyDeviceToHost);
    }

    ~MirroredArray() {
	if (owner) delete[] host;
	cudaFree(device);
    }

    T *host;
    T *device;
    size_t size;
    bool owner;
};

template<typename T>
class MirroredArrayDevice {
  public:   

    MirroredArrayDevice(T *data, size_t len) {
	size = len;
	host = data;
    owner = true;    
    cudaMalloc(&device, len*sizeof(T));
    copy();
    }

    void copy() {
	cudaMemcpy(device, host, size*sizeof(T), cudaMemcpyDeviceToDevice);
    }

    ~MirroredArrayDevice() {
	cudaFree(device);
    }

    T *device;
    T *host;
    size_t size;
    bool owner;
};

#endif
