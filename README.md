# Interactive Haptic Field

## A closed-loop image-to-haptics framework for accessible visual exploration

## Overview

The interactive haptic field (IHF) is a closed-loop image-to-haptics framework that converts image-derived geometry into position-dependent tactile cues during active fingertip exploration. IHF combines three-dimensional geometry reconstruction, low-latency fingertip trajectory prediction, spatially aligned digital-stimulus computation and hardware-specific actuation. The accompanying manuscript evaluates IHF using six human postures and three emojis with participants who had residual vision or blindness.

## Repository and manuscript correspondence

This is the companion implementation and data repository for the IHF manuscript. The public release covers the computational IHF pipeline and the research resources used to develop and evaluate it, including:

- three-dimensional content assets and digital-stimulus resources
- finger-trajectory data
- SAFTP and the comparison trajectory-prediction models
- model-training and evaluation scripts
- friction data used in the physical-rendering analysis

The participant-facing closed-loop system described in the manuscript integrates a Redmi Pad 2 Pro Android tablet, a custom infrared touch frame, an ITO-coated glass interface and a custom high-voltage electroadhesive driver. Device-specific Android application files, infrared-frame control files, PCB layouts, firmware and circuit-design files are not distributed through this public software repository. The evaluated hardware architecture and experimental configuration are described in the Methods and Supplementary Information.

## Repository structure

```text
IHF/
├── 3d_model/                       Three-dimensional assets and digital-stimulus resources
├── Friction_Data/                  Computed stimuli and experimental friction data
├── saftp/                          SAFTP and comparison-model code
└── finger_trajectory_dataset/  Recorded finger-trajectory data
```

### Three-dimensional content and digital stimuli

The `3d_model/` directory contains the point-cloud and digital-stimulus resources used for IHF content generation. The posture and emoji assets correspond to the content classes evaluated in the manuscript.

### Friction Data

This folder contains the digital stimuli and experimental friction measurement data used in this study.

## Swift Anchor-Free Finger Trajectory Prediction

The `saftp/` directory contains code for SAFTP and the comparison finger-trajectory prediction models.

### Dataset

The finger-trajectory dataset is available on Hugging Face at [ownt/IHF](https://huggingface.co/datasets/ownt/IHF). It contains three subdirectories:

- `finger_trajectory_straight_dec_2021`: straight finger trajectories
- `finger_trajectory_incline_jan_2024`: inclined finger trajectories
- `finger_trajectory_short_jan_2024`: short-distance finger trajectories

### Setup

1. Clone the repository and enter the SAFTP directory.

```bash
git clone --recursive https://github.com/whxueChris/IHF.git
cd IHF/saftp
```

2. Create and activate the environment.

```bash
conda env create -f environment.yaml
conda activate saftp
```

3. Download the trajectory dataset.

```bash
git lfs install
git clone https://huggingface.co/datasets/ownt/IHF data
```

### Usage

#### Train one model

```bash
python train_pure_decoder_dynamic.py \
  --data_dir ./data/finger_trajectory_dataset \
  --mode TDec
```

#### Evaluate one model

```bash
python train_pure_decoder_dynamic.py \
  --data_dir ./data/finger_trajectory_dataset \
  --mode TDec \
  --evaluate_only \
  --val_window_size_min 3 \
  --val_window_size_max 40
```

#### Train all models

```bash
bash train_all_models.sh
```

#### Evaluate all models

```bash
bash eval_all_models.sh
```

### Model modes

- `standard`: Swift Anchor-Free Finger Trajectory Prediction
- `mlp`: MLP-based trajectory prediction
- `autoregressive`: autoregressive trajectory prediction
- `TDec`: transformer decoder-only model

### Main parameters

- `--data_dir`: directory containing the finger-trajectory dataset
- `--mode`: model architecture
- `--window_size_min`: minimum training window size
- `--window_size_max`: maximum training window size
- `--val_window_size_min`: minimum validation window size
- `--val_window_size_max`: maximum validation window size
- `--teacher_forcing_ratio`: teacher-forcing ratio for autoregressive training
- `--evaluate_only`: evaluate without training

The training and evaluation shell scripts should be supplied with the same `--data_dir` used for the individual model commands.

## Plotting and latency measurement

`plot_image_analysis.py` provides model-performance visualization and inference-latency measurement for the trajectory-prediction models on the selected CPU or GPU.
