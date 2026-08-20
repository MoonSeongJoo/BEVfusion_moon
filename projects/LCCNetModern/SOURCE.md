LCCNet baseline source

Base implementation:
gitouni/Diffusion-Calib
IROS 2025
"Iterative Camera-LiDAR Calibration via Surrogate Diffusion Models"

The repository provides an unofficial PyTorch 2.x
reimplementation of LCCNet.

Local modifications:
1. Extracted LCCNet only.
2. Extracted ResnetEncoder only.
3. Replaced legacy correlation_cuda with pure-PyTorch
   correlation compatible with CUDA 12.x.
4. Replaced legacy Variable/.cuda() warp implementation
   with device-safe PyTorch implementation.
5. No Diffusion-Calib diffusion module is used.
