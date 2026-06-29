# Steady-Unsteady Paper Code

MATLAB analysis code for: *Temporal structure of task engagement reorganizes infra-slow BOLD dynamics*.

## System Requirements

- MATLAB R2021a or newer
- Required toolboxes: Signal Processing, Statistics and Machine Learning, Parallel Computing
- External: SPM12, CIFTI MATLAB utilities, Connectome Workbench

## Installation

1. Clone the repository
2. Configure all data and output paths in `config/project_paths.m`
3. Add dependencies to the MATLAB path:

Install time: <1 minutes if MATLAB and toolboxes are already available.

## Demo (no participant data required)

Runs a short Jansen-Rit/Balloon-Windkessel simulation to verify the installation:

```matlab
addpath('config')
addpath(fullfile(pwd, 'funcs', 'analysis'))
C = params_main();
C.jr.tmax = 30;
JR = jr_balloon_psd_three_states(C, 1);
```

Expected output: a `JR` struct with fields `psd.pre.rest`, `psd.pre.steady`, `psd.pre.unsteady`.  
Expected runtime: 1–5 minutes.

## Instructions for Use

1. Edit paths in `config/project_paths.m` and run `preprocess.m` to extract Glasser ROI time series from CIFTI BOLD files.
2. Run `main_analysis.m` section by section. Sections cover:
   - PSD power redistribution and sliding-window PSD
   - Frequency-resolved statistics and multiple-comparison correction
   - SVM classification and elastic-net behavior prediction
   - GLM residual comparison
   - Jansen-Rit + Balloon-Windkessel simulations and whole-brain network parameter sweeps

Full pipeline runtime: hours to days depending on permutation count and network sweep resolution. The JR demo alone takes 1–5 minutes.

