# Third-party compression algorithm notices

`compression_methods.py` adapts algorithmic structure from these official
research implementations:

- SparseGPT, IST-DASLab: https://github.com/IST-DASLab/sparsegpt
  (Apache License 2.0)
- GPTQ, IST-DASLab: https://github.com/IST-DASLab/gptq
  (Apache License 2.0)
- SmoothQuant, MIT HAN Lab, Copyright (c) 2022 MIT HAN Lab:
  https://github.com/mit-han-lab/smoothquant (MIT License)

The adaptations target arbitrary PyTorch `nn.Linear` submodules and use
fake-quantized arithmetic instead of the original model-specific CUDA kernels.
