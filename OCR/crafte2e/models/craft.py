"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import tensorflow as tf

# from basenet.vgg16_bn import vgg16_bn, init_weights

class double_conv(tf.keras.Model):
    def __init__(self, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(mid_ch, 1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(out_ch, 3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, x):
        x = self.conv(x)
        return x


class CRAFT(tf.keras.Model):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        # self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(512, 256)
        self.upconv2 = double_conv(256, 128)
        self.upconv3 = double_conv(128, 64)
        self.upconv4 = double_conv(64, 32)

        num_class = 2
        self.conv_cls = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding='same'), 
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, 3, padding='same'), 
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(16, 3, padding='same'), 
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(16, 1), 
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(num_class, 1)
        ])
        
    def call(self, sources):
        """ Base network """
        # sources = self.basenet(x)

        """ U network """
        y = tf.concat([sources[4], sources[3]], axis=3)
        y = self.upconv1(y)

        y = tf.image.resize(y, size=sources[2].shape[1:3])
        y = tf.concat([y, sources[2]], axis=3)
        y = self.upconv2(y)

        y = tf.image.resize(y, size=sources[1].shape[1:3])
        y = tf.concat([y, sources[1]], axis=3)
        y = self.upconv3(y)

        y = tf.image.resize(y, size=sources[0].shape[1:3])
        feature = tf.concat([y, sources[0]], axis=3)
        y = self.upconv4(feature)

        y = self.conv_cls(y)

        return y, feature

if __name__ == '__main__':
    pass