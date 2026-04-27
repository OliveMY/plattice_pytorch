"""Simple correctness tests for the CUDA permutohedral lattice extension.

These tests JIT-build the extension from the repository sources. They skip
cleanly when CUDA is not available.
"""

import math
import os
import unittest

import numpy as np
import torch
from torch.utils.cpp_extension import load


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSET_IMAGE = os.path.join(ROOT, "tests", "assets", "test_image.png")


def build_extension():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    return load(
        name="PLOO_test",
        sources=[
            os.path.join(ROOT, "permuto.cpp"),
            os.path.join(ROOT, "permutohedral.cu"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-gencode=arch=compute_86,code=sm_86",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def make_features(image, sigma_s=8.0, sigma_r=0.1):
    h, w, _ = image.shape
    ys, xs = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    return np.stack(
        [
            xs / sigma_s,
            ys / sigma_s,
            image[..., 0] / sigma_r,
            image[..., 1] / sigma_r,
            image[..., 2] / sigma_r,
        ],
        axis=-1,
    ).astype(np.float32)


def brute_force_bilateral(image, sigma_s=8.0, sigma_r=0.1):
    h, w, c = image.shape
    ys, xs = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    out = np.zeros_like(image)
    inv_2ss2 = 0.5 / (sigma_s * sigma_s)
    inv_2sr2 = 0.5 / (sigma_r * sigma_r)
    for y in range(h):
        for x in range(w):
            spatial = np.exp(-(((xs - x) ** 2 + (ys - y) ** 2) * inv_2ss2))
            color_delta = image - image[y, x]
            color = np.exp(-(np.sum(color_delta * color_delta, axis=-1) * inv_2sr2))
            weights = spatial * color
            denom = weights.sum()
            for ch in range(c):
                out[y, x, ch] = (weights * image[..., ch]).sum() / denom
    return out


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-12:
        return math.inf
    return 10.0 * math.log10(1.0 / mse)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PermutoLatticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ext = build_extension()

    def test_forward_backward_small_tensor(self):
        torch.manual_seed(7)
        features = torch.randn(16, 16, 5, device="cuda", dtype=torch.float32)
        values = torch.randn(16, 16, 3, device="cuda", dtype=torch.float32)

        weight, out = self.ext.forward(features.contiguous(), values.contiguous())
        self.assertEqual(tuple(weight.shape), (16, 16, 1))
        self.assertEqual(tuple(out.shape), (16, 16, 3))
        self.assertTrue(torch.isfinite(out).all().item())

        grad_values = self.ext.backward(features.contiguous(), torch.ones_like(out), weight)
        self.assertEqual(tuple(grad_values.shape), (16, 16, 3))
        self.assertTrue(torch.isfinite(grad_values).all().item())

    def test_runtime_dimensions(self):
        torch.manual_seed(11)
        for pd, vd in [(1, 1), (2, 4), (5, 3), (8, 2)]:
            features = torch.randn(12, 10, pd, device="cuda", dtype=torch.float32)
            values = torch.randn(12, 10, vd, device="cuda", dtype=torch.float32)
            weight, out = self.ext.forward(features.contiguous(), values.contiguous())
            self.assertEqual(tuple(weight.shape), (12, 10, 1))
            self.assertEqual(tuple(out.shape), (12, 10, vd))
            grad_values = self.ext.backward(features.contiguous(), torch.ones_like(out), weight)
            self.assertEqual(tuple(grad_values.shape), (12, 10, vd))

    def test_bilateral_image_sanity(self):
        from PIL import Image

        image = np.asarray(Image.open(ASSET_IMAGE).convert("RGB"), dtype=np.float32) / 255.0
        crop = image[48:80, 48:80].copy()
        features = torch.from_numpy(make_features(crop)).contiguous().cuda()
        values = torch.from_numpy(crop).contiguous().cuda()

        _weight, out = self.ext.forward(features, values)
        lattice = out.cpu().numpy()
        reference = brute_force_bilateral(crop)

        self.assertTrue(np.isfinite(lattice).all())
        self.assertGreaterEqual(float(lattice.min()), -1e-3)
        self.assertLessEqual(float(lattice.max()), 1.0 + 1e-3)
        self.assertGreater(psnr(lattice, reference), 18.0)
        self.assertGreater(psnr(lattice, reference), psnr(crop, reference))


if __name__ == "__main__":
    unittest.main()
