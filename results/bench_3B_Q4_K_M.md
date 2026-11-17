ggml_cuda_init: found 1 CUDA devices (Total VRAM: 8187 MiB):
  Device 0: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9, VMM: yes, VRAM: 8187 MiB
load_backend: loaded CUDA backend from $ROOT\bin\llama-b12473\ggml-cuda.dll
load_backend: loaded RPC backend from $ROOT\bin\llama-b12473\ggml-rpc.dll
load_backend: loaded CPU backend from $ROOT\bin\llama-b12473\ggml-cpu-zen4.dll
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | CUDA       |  99 |   1 |           pp512 |     5122.06 ± 531.63 |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | CUDA       |  99 |   1 |           tg128 |        107.28 ± 0.19 |

build: a49db739f (12473)
