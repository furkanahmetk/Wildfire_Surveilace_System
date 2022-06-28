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


## Google Drive folder which includes our custom mixed dataset and trained weights
[Google Drive of Mixed Dataset and pre-trained weights](https://drive.google.com/drive/folders/1yzvNfW-rJRnT51vGhIUvl5yQI0mwzn5x?usp=sharing)

## Installing Firenet Pre-Trained model
This is a pre-trained firenet model  which can also be used to test dataset. 
Following commands should be issued to install another pretrained model which was taken from [tobybreckon/fire-detection-cnn](https://github.com/tobybreckon/fire-detection-cnn):
```bash
cd ~/Wildfire_Surveilance_System/FireDetection/
./download-models.sh
```

## Links for Datasets that being tested and used to make our custom data set

[THE FLAME DATASET: AERIAL IMAGERY PILE BURN DETECTION USING DRONES (UAVS)(Shamsoshoara et al., 2021)](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)

[Effective Forest Fire Detection Data-set for Heterogeneous Wireless Multimedia Sensor Networks(Ever et al.,2020))](https://data.mendeley.com/datasets/g5nzp6j3bt/2)

[Dataset for Forest Fire Detection(Khan et al., 2020)](https://data.mendeley.com/datasets/gjmr63rz2r/1)

## Instructions to test with pre-trained models

Following paramaters should be set correctly to identify the path of pre-trained model and test data set and 
Edit main.py Line 136 and enter your weight path:
```
model.load("Model Path"),weights_only=True)
```
If comented uncommen Test Model part

Edit main.py Line 143 and enter your test path:

```
test_set = read_data("Test Path")
```
Make sure that your fire and nofire folder names match with catogories name
Catagories name in main.py Line 27
```
CATEGORIES = ["fire", "nofire"]
```
After clonning the repository and getting a decided weight, following commands should be issued to test with pretrained weights.
```bash
cd ~/Wildfire_Surveilance_System/FireDetection/
python3 main.py

```

## Training a model with a dataset
Open main.py

While consructing  model in line 95 set training true
```
model = construct_firenet(224, 224, training=True)
```

If comented uncomment Train Model part 
Make sure to comment out Load Model part

Set your training dataset path in line 101

```
train_set = read_data("Training path")

```

Make sure that your fire and nofire folder names match with catogories name

Catagories name in main.py Line 27

```
CATEGORIES = ["fire", "nofire"]
```
You can change test and validation split size in Line 117

```
(x_train, x_validate, y_train, y_validate) = train_test_split(x_train, y_train,test_size=0.20, random_state=42)
```

You can edit epeoch size in line 128

```
history = model.fit(x_train, y_train, 5,validation_set=(x_validate, y_validate))
```

After editing main.py, following commands should be issued to test the code.
```bash
cd ~/Wildfire_Surveilance_System/FireDetection/
python3 main.py

```

## Operating live Fire Detection test by using camera or video stream
Open main.py
Either train or load a model
Then uncomment Camera Model part
Make sure to comment Test Model part

Line 165 and 166 represents wide and height of our frame
```
rows = 224
cols = 224
```
At line 175 you need to decide your video capture method
Here youcan use system camera module or a video file
```
video = cv2.VideoCapture(0) # System camera module

or

video = cv2.VideoCapture("shortvideoplayback.mp4") # video file
```

After editing main.py, following commands should be issued to test the code.
```bash
cd ~/Wildfire_Surveilance_System/FireDetection/
python3 main.py
```

## Converting tensorflow to tensorflow lite to raspberry-pi implementation's performance improvement
You can convert pre trained firenet model to tenserflow lite model to increase the performance of fire detection in raspberry-pi.
Following commands should be issued to convert the  model which was taken from [tobybreckon/fire-detection-cnn]

```bash
cd ~/wildfire_Surveilance_System/FireDetection/converter/
python3 firenet-conversion.py
```
## Configuring power implementation
The figure below illustrates a reference diagram for the power system model of the drone: 
![2](https://user-images.githubusercontent.com/25657192/176139308-e0ea28df-a5ff-4cba-a25a-56b74c9fa249.PNG)

The general schematic of the solar charge controller is as shown below, according to (Swagatam Innovations, 2022):
![solar](https://user-images.githubusercontent.com/25657192/176139650-8cbb0001-57ae-4d63-8424-3dfb2e53cd4c.PNG)

## Automation implementation

Latest Version of Artupilot can be installed by following the [how to install Ardupilot guide](https://ardupilot.org/planner/docs/mission-planner-installation.html)

This should be set on ardupilot after installing it:
```bash

```
STL Architecture taken from Ardupilot (2021)(below)

![Ardupilotarch](https://user-images.githubusercontent.com/25657192/176140101-85656b71-2713-4a4a-8986-730cf755837d.png)

Automaion Levels
![auto](https://user-images.githubusercontent.com/25657192/176140149-d50d7f41-402c-4517-bae2-83f675124308.PNG)

Mission Planner screenshot of the Plan tab, showing retrieved data while mission is ongoing, 
![data](https://user-images.githubusercontent.com/25657192/176140215-bd0db0d0-e73f-447f-ba58-7c258636bc6d.PNG)

Mission Planner screenshot of the Plan tab, showing the test waypoints (green pins) which, the drone will move towards.(below)
![mp](https://user-images.githubusercontent.com/25657192/176140331-28386d52-d6b4-43f8-999a-870e301b2327.PNG)



## Communication implementation
For Telementry communication 3DR RF Module used to establish communication between pixhawk and ground unit. And Mavlink connection with cable used between raspberry-pi and pixhawk.

Simple showing of communication module (below)
![communicationmodule](https://user-images.githubusercontent.com/20406719/176122442-7fcb8c38-a674-432e-bec2-940033209053.png)


## References

Alireza Shamsoshoara, Fatemeh Afghah, Abolfazl Razi, Liming Zheng, Peter Fulé, Erik Blasch. (2020). The FLAME dataset: Aerial Imagery Pile burn detection using drones (UAVs). IEEE Dataport. https://dx.doi.org/10.21227/qad6-r683

Allison, R., Johnston, J., Craig, G., & Jennings, S. (2016). Airborne Optical and Thermal Remote Sensing for Wildfire Detection and Monitoring. Sensors, 16(8), 1310. https://doi.org/10.3390/s16081310

Alhadi, S., Rianmora, S., & Phlernjai, M. (2021). Conceptual design and analysis of small power stations for supporting unmanned aerial vehicle (UAV) deployment. Engineering Journal, 25(8), 51-71. doi:10.4186/ej.2021.25.8.51

Ardupilot. (2021). Retrieved from Mission Planner Home: https://ardupilot.org/planner/index.html 

Ardupilot. (2021). Retrieved from SITL Simulator (Software in the Loop): https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html

Ever, Enver; Yatbaz, Hakan Yekta ; Kizilkaya, Burak; Yazici, Adnan (2020), “Effective Forest Fire Detection Data-set for Heterogeneous Wireless 

Multimedia Sensor Networks”, Mendeley Data, V2, doi: 10.17632/g5nzp6j3bt.2

Facts + Statistics: Wildfires | III. (2021). Insurance Information Institute. https://www.iii.org/fact-statistic/facts-statistics-wildfires 

Gholamnia, K., Gudiyangada Nachappa, T., Ghorbanzadeh, O., & Blaschke, T. (2020). Comparisons of Diverse Machine Learning Approaches for Wildfire 

Susceptibility Mapping. Symmetry, 12(4), 604. https://doi.org/10.3390/sym12040604

Ghorbanzadeh, O., Blaschke, T., Gholamnia, K., & Aryal, J. (2019). Forest Fire Susceptibility and Risk Mapping Using Social/Infrastructural Vulnerability and Environmental Variables. Fire, 2(3), 50. https://doi.org/10.3390/fire2030050

Grepow Blog. (2020, July 21). FPV drone flight time: How to calculate?     https://www.grepow.com/blog/how-to-calculate-fpv-drone-flight-time/

Khan, Ali; Hassan, Bilal (2020), “Dataset for Forest Fire Detection”, Mendeley Data, V1, doi: 10.17632/gjmr63rz2r.1

Lee, T., Mckeever, S., & Courtney, J. (2021, June 17). Flying Free: A Research Overview of Deep Learning in Drone Navigation Autonomy. MDPI. https://www.mdpi.com/2504-446X/5/2/52/htm

Mark A. Finney, C. W. (2012, 12 19). FARSITE. Retrieved from fs.usda: https://www.fs.usda.gov/rmrs/tools/farsite

Pandian, S. (2022, February 17). K-fold cross validation technique and its essentials. Analytics Vidhya. Retrieved June 19, 2022, from https://www.analyticsvidhya.com/blog/2022/02/k-fold-cross-validation-technique-and-its-essentials/ 

Pixhawk series. (2021, 11 23). Retrieved from docs.px4: https://docs.px4.io/master/en/flight_controller/pixhawk_series.html

Precision Landing and Loiter with IR-LOCK. (2021). Retrieved from Ardupilot: https://ardupilot.org/copter/docs/precision-landing-with-irlock.html

Singh, R. (2021, 1 25). easy way to integrate ai with ardupilot oak-d. Retrieved from discuss.ardupilot: https://discuss.ardupilot.org/t/easy-way-to-integrate-ai-with-ardupilot-oak-d-part-1/79306

Swagatam Innovations. (2022, June 1). Best Electronic Circuit Projects. Homemade Circuit Projects. https://www.homemade-circuits.com/

T. Artés, A. C. (2016). Large Forest Fire Spread Prediction: Data and Computational Science. Procedia Computer Science, 909-918.

WindNinja. (n.d.). Retrieved from firelab: https://www.firelab.org/project/windninja

Yiu, T. (2021, September 29). Understanding Random Forest - Towards Data Science. Medium. https://towardsdatascience.com/understanding-random-forest-58381e0602d2

Dunnings, A. J., & Breckon, T. P. (2018). Experimentally Defined Convolutional Neural Network Architecture Variants for Non-Temporal Real-Time Fire 

Detection. 2018 25th IEEE International Conference on Image Processing (ICIP). https://doi.org/10.1109/icip.2018.8451657

## Acknowledgements

Wights taken from github repository https://github.com/tobybreckon/fire-detection-cnn

