# Compute and Environment

Generated from `artifacts/system/environment.json` captured at 2026-09-27T07:45:31.548958+00:00.

- Python: `3.12.10` in the qualified local project virtual environment
- PyTorch/CUDA: `2.14.0+cu130` / runtime `13.0`
- GPU: `NVIDIA GeForce RTX 4090 Laptop GPU` with 15.99 GiB
- CuPy/runtime: `14.2.0` / `13020`
- NumPy/SciPy/QuantLib: `2.5.3` / `1.18.1` / `1.43`

CUDA FP64 smoke tests passed. Research workloads recorded on GPU include batched Heston CF,
Heston Monte Carlo, and synthetic truth generation. The measured small CF batch speedup was
0.711x (slower than CPU); the medium batch speedup was
35.053x. GPU benefit is workload-size dependent.

The final benchmark manifest records profile `research` on `cuda` across 4 workload families. Measured CPU/GPU speedups range from 0.1765x to 121.9785x.
The N10 precision study measured maximum FP64/FP32 absolute errors of 4.263e-14/5.876e-05; FP64 remains mandatory.

## Reproducibility

Primary full run:
`.\.venv\Scripts\python.exe -m derivguard full --profile research --device cuda --resume`

Resume uses the same command. Machine-readable outputs record seeds, backends, timestamps, hashes,
and frozen design identifiers. The 20 GiB disk reserve remains mandatory.
