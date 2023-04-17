from classes.FallDetectionModel import FallDetectionModel
from classes.YoloV5 import YoloV5

# Import necessary libraries
import tensorflow as tf
import numpy as np
import cv2
import os

# Load the YOLOv5 model
yolov5 = YoloV5('classes/yolov5s.yaml')
yolov5.load_weights('classes/yolov5s.pt')

# Load the Fall Detection model
fall_detection_model = FallDetectionModel(yolov5)
fall_detection_model.model().summary()
