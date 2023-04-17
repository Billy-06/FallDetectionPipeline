from classes.FallDetectionModel import FallDetectionModel
from classes.YoloV5 import YoloV5

# Import necessary libraries
import tensorflow as tf
import numpy as np
import cv2
import os
from os import path

# Use path to load the YoloV5 config file and weights file
yolo_dir = path.join(path.dirname(__file__), 'classes')
yolo_cfg = path.join(yolo_dir, 'yolov5s.yaml')
yolo_weights = path.join(yolo_dir, 'yolov5.pt')

# Check if the YoloV5 config file and weights file exist
if not path.exists(yolo_cfg):
    raise Exception(f"File not found: {yolo_cfg}")
if not path.exists(yolo_weights):
    raise Exception(f"File not found: {yolo_weights}")


# Load the YOLOv5 model
yolov5 = YoloV5(yolo_cfg)
yolov5.load_weights(yolo_weights)

# Load the Fall Detection model
fall_detection_model = FallDetectionModel(yolo_weights, yolo_cfg)
fall_detection_model.model().summary()

print(f"{YoloV5.__name__} weights: {yolov5.weights}")
print(f"{FallDetectionModel.__name__} weights: {fall_detection_model.yolov5.weights}")

print('Done')
