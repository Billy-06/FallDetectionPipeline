# Import necessary libraries
import tensorflow as tf
import numpy as np
import cv2
import os


class FallDetectionModel(tf.keras.Model):
    def __init__(self, yolov5_weights):
        super(FallDetectionModel, self).__init__()
        self.yolov5 = YOLOv5('yolov5s.yaml')
        self.yolov5.load_weights(yolov5_weights)
        self.yolov5.trainable = False
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs):
        x = self.yolov5(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)

    def model(self):
        x = tf.keras.Input(shape=(640, 640, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def get_layer_output(self, inputs, layer_name):
        intermediate_layer_model = tf.keras.Model(
            inputs=self.model().inputs, outputs=self.model().get_layer(layer_name).output)
        return intermediate_layer_model(inputs)

    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False, layer_range=None):
        return super().summary(line_length, positions, print_fn, expand_nested, show_trainable, layer_range)
