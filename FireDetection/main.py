################################################################################

import cv2
import os
import sys
import math
import time
from datetime import datetime

################################################################################
import numpy as np
import tensorflow

import tflearn
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression
from pymavlink import mavutil

from sklearn.model_selection import train_test_split

################################################################################
from tqdm import tqdm

CATEGORIES = ["fire", "nofire"]


def construct_firenet(x, y, training=False):
    # Build network as per architecture in [Dunnings/Breckon, 2018]

    network = tflearn.input_data(shape=[None, y, x, 3], dtype=tensorflow.float32)

    network = conv_2d(network, 64, 5, strides=4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 128, 4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 256, 1, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = fully_connected(network, 4096, activation='tanh')
    if (training):
        network = dropout(network, 0.5)

    network = fully_connected(network, 4096, activation='tanh')
    if (training):
        network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='softmax')

    # if training then add training hyperparameters

    if (training):
        network = regression(network, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)

    # constuct final model

    model = tflearn.DNN(network, checkpoint_path='firenet',
                        max_checkpoints=1, tensorboard_verbose=2)

    return model


def read_data(directory):
    data_set = []
    for category in CATEGORIES:
        path = os.path.join(directory, category)  # create path
        class_num = CATEGORIES.index(category)  # get the classification

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path, img))  # convert to array
                new_array = cv2.resize(img_array, (224, 224), cv2.INTER_AREA)  # resize to normalize data size
                data_set.append([new_array, CATEGORIES[class_num]])  # add this to our training_data
            except Exception as e:
                print(e)
    return data_set


def run():

    ################################################################################
    # Construct model

    model = construct_firenet(224, 224, training=False)
    print("Constructed FireNet ...")

    ################################################################################
    # Train Model
    """
    train_set = read_data("Dataset1/Training")
    x_train = []
    y_train = []
    x_data = []
    y_data = []
    x_validate = []
    y_validate = []
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    for features, label in train_set:
        x_train.append(features)
        y_train.append(label)

    (x_train, x_validate, y_train, y_validate) = train_test_split(x_train, y_train,test_size=0.20, random_state=42)

    x_train = np.array(x_train)
    x_validate = np.array(x_validate)

    y_train = label_encoder.fit_transform(y_train)
    y_train = y_train.reshape(len(y_train), 1)
    y_train = onehot_encoder.fit_transform(y_train)

    y_validate = label_encoder.fit_transform(y_validate)
    y_validate = y_validate.reshape(len(y_validate), 1)
    y_validate = onehot_encoder.fit_transform(y_validate)

    history = model.fit(x_train, y_train, 5,validation_set=(x_validate, y_validate))

    model.save("last_model.tfl")
    """
    ################################################################################
    # Load Model

    model.load(os.path.join("models/FireNet", "firenet"),weights_only=True)
    print("Loaded CNN network weights ...")

    ################################################################################
    #Test Model

    """
    test_set = read_data("Dataset2/test")
    x_test = []
    y_test = []
    for features, label in test_set:
        x_test.append(features)
        y_test.append(label)

    x_test = np.array(x_test)
    y_test = label_encoder.fit_transform(y_test)
    y_test = y_test.reshape(len(y_test), 1)
    y_test = onehot_encoder.fit_transform(y_test)

    print("Evaluating network...")
    predictions = model.predict(x_test)
    print(classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=CATEGORIES))
    """

    ################################################################################
    # Camera Model

    # network input sizes

    rows = 224
    cols = 224

    # display and loop settings

    windowName = "Live Fire Detection - FireNet CNN";
    keepProcessing = True;

    # load video file from first command line argument

    video = cv2.VideoCapture("shortvideoplayback.mp4")
    print("Loaded video ...")

    # create window

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);

    # get video properties

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_time = round(1000 / fps);
    logfile = open("firelog.txt",'w')

    gcs_conn = mavutil.mavlink_connection('tcp:localhost:15795', input=False)
    gcs_conn.wait_heartbeat()
    print("Heartbeat from system (system %u component %u)" % (gcs_conn.target_system, gcs_conn.target_system))


    pass_count = 0;

    while (keepProcessing):

        # start a timer (to see how long processing and display takes)

        start_t = cv2.getTickCount();
        

        # get video frame from file, handle end of file
        start_time = time.monotonic()
        ret, frame = video.read()
        if not ret:
            print("... end of video file reached");
            break;

        # re-size image to network input size and perform prediction

        small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)

        # perform prediction on the image frame which is:
        # - an image (tensor) of dimension 224 x 224 x 3
        # - a 3 channel colour image with channel ordering BGR (not RGB)
        # - un-normalised (i.e. pixel range going into network is 0->255)

        output = model.predict([small_frame])

        # label image based on prediction

        if round(output[0][0]) == 1:
            if pass_count <= 10:
                pass_count += 1
        else:
            if pass_count >= 0:
                pass_count -= 1
               
        if pass_count > 5 :

            gcs_msg = b"fire detected."
            gcs_conn.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_CRITICAL, gcs_msg)
            print(gcs_msg)
            now = datetime.now()
            logfile.write(now.strftime("%H:%M:%S"))
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 50)
            cv2.putText(frame, 'FIRE', (int(width / 16), int(height / 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 10, cv2.LINE_AA);
        else:
            gcs_msg = b"clear."
            gcs_conn.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_CRITICAL, gcs_msg)
            print(gcs_msg)
            cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 50)
            cv2.putText(frame, 'CLEAR', (int(width / 16), int(height / 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 10, cv2.LINE_AA);
        

        # stop the timer and convert to ms. (to see how long processing and display takes)

        stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000;

        # image display and key handling

        cv2.imshow(windowName, frame);
        elapsed_sec = (time.monotonic() - start_time)
        fps = 1/elapsed_sec
        print("\nFPS:")
        print(fps)
        
        # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)
        key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF;
        if (key == ord('x')):
            keepProcessing = False;
        elif (key == ord('f')):
            cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
    logfile.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
