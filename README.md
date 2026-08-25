# Fused Softmax and SGEMM

Tayte Choudhury

CUDA C++ implementations of online softmax and naive, shared-memory tiled,
register-blocked matrix multiplication. The executable checks softmax against
a double-precision CPU reference and SGEMM against cuBLAS pedantic FP32.

Build from a Visual Studio x64 developer prompt with CUDA 12.9 or compatible:

```sh
nvcc -O3 -arch=sm_86 kernels.cu -lcublas -o benchmark.exe
benchmark.exe > results.csv
```

Three warmups, twenty CUDA-event repetitions, transfers excluded. Shapes include
non-tile-aligned 65-square matrices. See results.csv for actual RTX 3070 measurements.
Softmax effective bandwidth counts 8 bytes per element, although this kernel reads
inputs twice. It uses one fused kernel launch, not a one-read memory algorithm.
Only finite inputs are supported. NaN and infinity semantics are not implemented.

These new measurements do not reproduce the historical resume figures. No Nsight
profile or PyTorch extension is included. Run `ncu ./benchmark.exe` in an environment
with Nsight Compute installed to collect a separate profile.

References: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

The public/ folder contains only the portfolio page and selected results. The repository is currently public. A future private repository can host only public/ on Render.


Portfolio: https://t8c4.com/projects/cuda-kernels.html

Repository: https://github.com/t8dahgr8/tayte-cuda-kernels


The complete original folder layout, public presentation and Render configuration are included in project-bundle.zip. Extract that archive before following instructions referring to public/.
