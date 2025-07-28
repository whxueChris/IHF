# IHF-preview

## Interactive Haptic Fields Enable Touch Exploration of Digital Images


**Abstract:** Haptic interaction has emerged as a fundamental modality in digital communication, with haptic rendering serving as the bridge between digital stimuli and perceptible physical haptic feedback. For individuals who are blind or partially sighted, this form of feedback is essential for accessing and understanding digital information. Current haptic rendering techniques, however, lack a universal and dynamically responsive strategy for diverse and efficient interactions. Here, we introduce the concept of   interactive haptic field (IHF), a generalized haptic information generation framework that adaptively renders haptic information from arbitrary digital images.   This framework formulates haptic generation as a composite mapping function that integrates digital stimuli from target surface characteristics, actuation signals, constitutive relations of different haptic interfaces, and user perceptual models. We validated the  framework to render digital portrait images across three distinct haptic devices. Experimental results confirm that our approach accurately reproduces physical stimuli on different devices, enabling users to effectively perceive the digital portraits by touch.  With the ability to generate haptic information from any digital image on both current and future devices, IHF has the potential to drive full digital inclusion, enhance cultural accessibility, and transform user interaction across a wide range of applications.


## Instructions for use

This repository contains source code and data for IHF haptic information rendering of human posture on TanvasTouch and iOS devices.

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


