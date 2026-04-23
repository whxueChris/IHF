# IHF-preview

## Bridging the images and touch through an interactive haptic field system


**Abstract:** As digital information becomes pervasive, haptic interaction is emerging as an important interface for human–machine communication, particularly for people who are blind or partially sighted, for whom tactile feedback is crucial to accessing visual content. However, existing haptic rendering approaches are typically tailored to specific devices or content types, and lack a common computational representation that can support both an optimized haptic system and deployment, with controlled trade-offs, across diverse imagery and hardware platforms. Here we introduce the interactive haptic field (IHF), a generalized computational framework that converts arbitrary images into a device-agnostic haptic representation, which can then be realized in optimized haptic systems and deployed across multiple feedback modalities and devices. IHF combines image-based 3D geometry reconstruction with a low-latency, anchor-free finger trajectory prediction model to generate localized, path-dependent digital stimuli, which are then mapped onto device-specific actuation signals. We implement an integrated, closed-loop image-to-haptics system that co-designs the IHF framework with an electroadhesion-based surface haptic display, and we further demonstrate deployment on a smartphone vibration interface, achieving image-recognition success rates above 70% in tasks involving human posture classification, pathology case differentiation and emoji identification. By decoupling haptic representation from specific devices and content domains, while providing a high-fidelity reference implementation and demonstrating portability to commodity hardware, IHF offers a scalable strategy for tactile displays of images, with implications for accessible interfaces in education, healthcare, robotics and emotion interaction.


## Instructions for use

This repository contains source code and data for IHF haptic information rendering of human posture and pathology cases on TanvasTouch and iOS devices.

### TanvasTouch
#### System requirements

- Test on **Windows 11**
- **.NET Framework 4.8** or later / **.NET Core 3.1** or later
- Python 3, Visual Studio 2019/2022
- Required NuGet Packages: `Tanvas.TanvasTouch`, `Tanvas.TanvasTouch.WpfUtilities`,`Microsoft.ML.OnnxRuntime`, `Microsoft.ML.OnnxRuntime.Tensors`, `NumSharp`, `MathNet.Numerics`
- Required Python Packages: `pywin32`, `numpy`, `matplotlib`, `opencv-python`, `scipy`, `scikit-learn`, `argparse`
- TanvasTouch Engine,  please check https://tanvas.co/resources/tanvastouch-basics for more details



#### Folder structure

- The file is structured as follows:

```
Tanvas
├── Dancing/                (Point cloud data of dancing posture)
|   |-- cluster_points_1.txt 
    |-- cluster_points_2.txt 
    |-- ...
├── Gymnastics/             (Point cloud data of gymnastics posture)
├── Hands up/               (Point cloud data of hands posture)
├── Horse stance/           (Point cloud data of  horse stance posture)
├── Stand/                  (Point cloud data of stand posture)
├── Yoga/                   (Point cloud data of yoga postures)
├── Pretraining/                   (Point cloud data of pretraining postures)
├── TanvasTouch_Posture/    (Visual Studio project for haptic rendering on TanvasTouch.It captures real-time finger trajectories, predicts future movements, receives digital stimuli from Python, and generates corresponding physical stimuli)
├── recive.py          (Script for processing trajectory date, and compute digital stimuli)

```

#### Steps to run

1. Run the python script: `python receive.py`
2. Open the Visual Studio solution (.sln) file and run the project.



### iOS device


#### Build and deploy the iOS app through Xcode

1. Open Xcode and create a new iOS App project.
2. Copy the contents of `IOS_PoseDetector/PoseDetector/` into your project folder.
3. In Xcode, import all `.swift` files and resources from `Model/`, `View/`, and `Data/`, and drag in the `.mlmodel` file (Xcode will auto-compile it). Merge the provided `Info.plist` to include necessary permissions.
4. Set up the initial view controller (via Storyboard or SwiftUI), then connect your iOS device, and build the project.
5. Export the `.ipa` and install it on your iOS device.

### Friction Data
This folder contains the digital stimuli and experimental friction measurement data used in this study.

### Swift Anchor-Free Finger Trajectory Prediction  (SAFTP)

This repository contains code for finger trajectory prediction models using various transformer-based architectures.

#### Dataset

The dataset is available on HuggingFace: [<https://huggingface.co/datasets/ownt/IHF>](<https://huggingface.co/datasets/ownt/IHF>)

It contains three subdirectories:
- `finger_trajectory_straight_dec_2021`: Straight finger trajectories
- `finger_trajectory_incline_jan_2024`: Inclined finger trajectories
- `finger_trajectory_short_jan_2024`: Short distance finger trajectories

#### Setup

1. Download our repository and open the saftp

```angular2html
git clone --recursive https://github.com/xwhkkk/IHF.git
cd saftp
```

2. Install saftp and its dependencies, but the package version is not strictly required.
```angular2html
conda env create -f environment.yaml
conda activate saftp
```

3. Download the dataset from HuggingFace:

```angular2html
git lfs install
git clone https://huggingface.co/datasets/ownt/IHF data
```


#### Usage

##### Training a model

```angular2html
python train_pure_decoder_dynamic.py --data_dir ./data/tanvastouch_finger_trajectory --mode TDec
```

##### Evaluating a model

```angular2html
python train_pure_decoder_dynamic.py --data_dir ./data/tanvastouch_finger_trajectory --mode TDec --evaluate_only --val_window_size_min 3 --val_window_size_max 40
```

##### Training all models

```angular2html
bash train_all_models.sh
```


##### Evaluating all models

```angular2html
bash eval_all_models.sh
```

## Model Modes
- `standard`: Swift Anchor-Free Finger Trajectory Prediction
- `mlp`: MLP-based Decoder for Finger Trajectory Prediction
- `autoregressive`: Autoregressive Decoder for Finger Trajectory Prediction
- `TDec`: Pure Transformer Decoder-only model

## Parameters

- `--data_dir`: Directory containing the finger trajectory datasets
- `--mode`: Model architecture to use
- `--window_size_min`: Minimum window size for training
- `--window_size_max`: Maximum window size for training
- `--val_window_size_min`: Minimum validation window size
- `--val_window_size_max`: Maximum validation window size
- `--teacher_forcing_ratio`: Ratio for teacher forcing in autoregressive training
- `--evaluate_only`: Only evaluate the model without training

The shell scripts (train_all_models.sh and eval_all_models.sh) should also be updated to include the --data_dir parameter when calling the Python scripts.

## Plot and Measure Latency

-`plot_image_analysis.py`   (Benchmarking and visualization tool for comparing performance metrics of different trajectory prediction models. Measures inference latency on CPU/GPU and generates comparison plots of model performance across different sequence lengths)
