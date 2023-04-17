from os import path
import tensorflow as tf
import numpy as np

"""
    > This class is used to define the YoloV5 model
    > The YoloV5 model is defined in the YoloV5.yaml file
    > The YoloV5.yaml file is converted to TensorFlow format
    File is downloaded from
    https://raw.githubusercontent.com/ultralytics/yolov5/master/models/yolov5s.yaml

"""


class YoloV5(tf.keras.Model):
    def __init__(self, cfg_file):
        super(YoloV5, self).__init__()
        self.cfg_file = cfg_file
        self.model = self.build_model()

    def __repr__(self):
        return self.model.__repr__()

    def build_model(self):
        # Load the YoloV5 model config file ( yolov5s.yaml )
        with open(self.cfg_file, 'r') as f:
            cfg = f.read()

        # Convert the YoloV5 config to TensorFlow format
        model = tf.keras.models.Sequential()
        idx = 0
        for line in cfg.split('\n'):
            if line.startswith('-'):
                if 'convolutional' in line:
                    filters = int(cfg.split('\n')[idx+1].split('=')[1].strip())
                    kernel_size = int(
                        cfg.split('\n')[idx+2].split('=')[1].strip())
                    stride = int(cfg.split('\n')[idx+3].split('=')[1].strip())
                    pad = int(cfg.split('\n')[idx+4].split('=')[1].strip())
                    model.add(tf.keras.layers.Conv2D(
                        filters=filters, kernel_size=kernel_size, strides=stride, padding='same' if pad else 'valid'))
                    if 'batch_normalize' in cfg.split('\n')[idx+5]:
                        model.add(tf.keras.layers.BatchNormalization())
                    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
                elif 'maxpool' in line:
                    size = int(cfg.split('\n')[idx+1].split('=')[1].strip())
                    stride = int(cfg.split('\n')[idx+2].split('=')[1].strip())
                    model.add(tf.keras.layers.MaxPooling2D(
                        pool_size=size, strides=stride))
                elif 'route' in line:
                    layers = [int(x) for x in line.split('=')[1].split(',')]
                    if len(layers) == 1:
                        model.add(model.layers[layers[0]])
                    else:
                        concat_layers = [model.layers[layer]
                                         for layer in layers]
                        model.add(tf.keras.layers.Concatenate(concat_layers))
                elif 'shortcut' in line:
                    from_layer = int(line.split('=')[1].strip())
                    model.add(tf.keras.layers.Add())
                elif 'upsample' in line:
                    stride = int(line.split('=')[1].strip())
                    model.add(tf.keras.layers.UpSampling2D(
                        size=(stride, stride)))
            idx += 1

        return model

    def load_weights(self, weights_file):
        # Load the YoloV5 model weights
        with open(weights_file, 'rb') as f:
            major, minor, revision, seen, _ = np.fromfile(
                f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)

        # Assign the YoloV5 model weights to the TensorFlow model
        idx = 0
        for layer in self.model.layers:
            if 'conv' in layer.name:
                if 'batch_normalization' in layer.name:
                    bn_layer = layer
                    bn_layer.gamma.assign(
                        weights[idx:idx+bn_layer.gamma.shape[0]])
                    idx += bn_layer.gamma.shape[0]
                    bn_layer.beta.assign(
                        weights[idx:idx+bn_layer.beta.shape[0]])
                    idx += bn_layer.beta.shape[0]
                    bn_layer.moving_mean.assign(
                        weights[idx:idx+bn_layer.moving_mean.shape[0]])
                    idx += bn_layer.moving_mean.shape[0]
                    bn_layer.moving_variance.assign(
                        weights[idx:idx+bn_layer.moving_variance.shape[0]])
                    idx += bn_layer.moving_variance.shape[0]
                else:
                    conv_layer = layer
                    conv_layer.kernel.assign(weights[idx:idx+conv_layer.kernel.shape[0]
                                                     * conv_layer.kernel.shape[1]*conv_layer.kernel.shape[2]*conv_layer.kernel.shape[3]].reshape(
                        conv_layer.kernel.shape[3], conv_layer.kernel.shape[2], conv_layer.kernel.shape[0], conv_layer.kernel.shape[1]))
                    idx += conv_layer.kernel.shape[0] * \
                        conv_layer.kernel.shape[1] * \
                        conv_layer.kernel.shape[2]*conv_layer.kernel.shape[3]
                    conv_layer.bias.assign(
                        weights[idx:idx+conv_layer.bias.shape[0]])
                    idx += conv_layer.bias.shape[0]

    def call(self, inputs):
        return self.model(inputs)

    def model(self):
        x = tf.keras.Input(shape=(640, 640, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
