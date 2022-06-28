# Wildfire Surveilace System
Wildfire suveilance system with an autonomous UAV(Drone), which uses Custome CNN firenet model with raspberry-pi implementation. This repository also includes all module details like power module communication and automation modules.

Tested using Python 3.7.x, [TensorFlow 1.15](https://www.tensorflow.org/install/), [TFLearn 0.3.2](http://tflearn.org/) and [OpenCV 3.x / 4.x](http://www.opencv.org) (requires opencv extra modules - ximgproc module for superpixel segmentation)

## Abstract

In this repository we worked on our multidisciplinary final year project.Our final year multidisciplinary team project is an autonomous wildfire surveillance drone running on renewable energy. The goal of this drone is to detect and announce when it detects a wildfire to alarm authorities before the wildfire causes any significant damage. We are using an Arducopter to run Ardupilot. It uses GPS and radio frequency to control the drone. The detection of fire happens in the drone. We are using a raspberry to run our CNN detection algorithm. The CNN algorithm uses a simplified version of the FireNet with TensorFlow.  The drone communicates to a ground station sending various types of information about the drone’s state. The drone’s battery is charged using solar panels. Fire detection using image processing and machine learning. Simulation and testing have also been done through Mission Planner, which will also be used as the GUI for the drone’s ground station.
## Design Charts and Diagrams

General Operation flow diagram (below)

![GeneralOperationFlogDiagram](https://user-images.githubusercontent.com/25657192/176115243-e8248eb7-3056-4380-9336-af4b90912db7.png)

![](https://github.com/furkanahmetk/Wildfire_Surveilace_System/GeneralOperationFlogDiagram.png)

Communication overall design structure (below)

![CommunicationDesign](https://user-images.githubusercontent.com/25657192/176115417-be704ebd-4a9c-493a-b534-6c0ae4250d24.png)

![](https://github.com/furkanahmetk/Wildfire_Surveilace_System/CommunicationDesign.png)

Firenet model structure (below)

![StructureOfFirenetModel](https://user-images.githubusercontent.com/25657192/176115446-e21dd098-fe7d-45d7-be06-f24e2b0e5ca6.png)

![](https://github.com/furkanahmetk/Wildfire_Surveilace_System/StructureOfFirenetModel.png)


## Clonning the repository & installing required packages for fire detection

Following commands should be issued to clone repository and install required packages on terminal
```bash
git clone https://github.com/furkanahmetk/Wildfire_Surveilace_System.git
sudo apt install pip3
pip3 install opencv-contrib-python
pip3 install tensorflow==1.15.4
pip3 install tflearn
```

On Development process of fire detection module PyCharm has been used with convo virtual environment. To set the convo environment for pycharm, following instructions should be applied if you are planning to use pycharm too :

To install pycharm visit [Pycharm Official Website](https://www.jetbrains.com/pycharm/download/#section=linux)




## Google Drive folder which includes our custom mixed dataset and pre trained weights
[Google Drive of Mixed Dataset and pre-trained weights](https://drive.google.com/drive/folders/1yzvNfW-rJRnT51vGhIUvl5yQI0mwzn5x?usp=sharing)

following commands should be issued to install another pretrained model :
```bash
cd ~/Wildfire_Surveilance_System/FireDetection/
./download-models.sh
```

## Links for Datasets that being tested and used to make our custom data set

[THE FLAME DATASET: AERIAL IMAGERY PILE BURN DETECTION USING DRONES (UAVS)(Shamsoshoara et al., 2021)](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)

[Effective Forest Fire Detection Data-set for Heterogeneous Wireless Multimedia Sensor Networks(Ever et al.,2020))](https://data.mendeley.com/datasets/g5nzp6j3bt/2)

[Dataset for Forest Fire Detection(Khan et al., 2020)](https://data.mendeley.com/datasets/gjmr63rz2r/1)

## Instructions to test pre-trained models

Following paramaters should be set correctly to identify the path of pre-trained model and test data set
main.py Line 144:
```bash

```

## Converting tensorflow to tensorflow lite to raspberry-pi implementation's performance improvement

Following paramaters should be set correctly to identify the path of pre-trained model and test data set
main.py Line 144:
```bash

```

Following commands should be issued to test pretrained weights, if you already clonned the repository and downloaded pretrained models you may ignore first related commands.

```bash
git clone https://github.com/furkanahmetk/Wildfire_Surveilace_System.git
cd ~/Wildfire_Surveilance_System/FireDetection/
./download-models.sh


```

## Training a model with a dataset

```bash

```

## Operating live Fire Detection test by using camera or video stream

```bash

```
## Configuring power implementation

## Automation implementation

## Communication implementation

## References

## Acknowledgements


