import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
# from models.deformable_conv import DeformableConvLayer
from models.layers import ConvOffset2D


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)

############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

class IdentityBlock(KL.Layer):

    def __init__(self, kernel_size, filters, stage, block, use_bias=True):
        super(IdentityBlock, self).__init__()
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.a = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',use_bias=use_bias)
        self.a_batchnorm = BatchNorm(name=bn_name_base + '2a')
        self.a_relu = KL.Activation('relu')

        self.b = KL.Conv2D(nb_filter2, (kernel_size, 1), padding='same',
                           name=conv_name_base + '2b', use_bias=use_bias)
        self.b_batchnorm = BatchNorm(name=bn_name_base + '2b')
        self.b_relu = KL.Activation('relu')

        self.c = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                           use_bias=use_bias)
        self.c_batchnorm = BatchNorm(name=bn_name_base + '2c')

        self.add = KL.Add()
        self.res_relu = KL.Activation('relu', name='res' + str(stage) + block + '_out')

    def call(self, input_tensor, train_bn=None):
        """The identity_block is the block that has no conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            use_bias: Boolean. To use or not use a bias in conv layers.
            train_bn: Boolean. Train or freeze Batch Norm layers
        """
        x = self.a(input_tensor)
        x = self.a_batchnorm(x, training=train_bn)
        x = self.a_relu(x)

        x = self.b(x)
        x = self.b_batchnorm(x, training=train_bn)
        x = self.b_relu(x)

        x = self.c(x)
        x = self.c_batchnorm(x, training=train_bn)

        x = self.add([x, input_tensor])
        x = self.res_relu(x)
        return x

class ConvBlock(KL.Layer):

    def __init__(self, kernel_size, filters, stage, block,
                 strides=(2, 1), use_bias=True):
        super(ConvBlock, self).__init__()
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.a2 = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                            name=conv_name_base + '2a', use_bias=use_bias)
        self.a2_batchnorm = BatchNorm(name=bn_name_base + '2a')
        self.a2_relu = KL.Activation('relu')

        self.b2 = KL.Conv2D(nb_filter2, (kernel_size, 1), padding='same',
                            name=conv_name_base + '2b', use_bias=use_bias)
        self.b2_batchnorm = BatchNorm(name=bn_name_base + '2b')
        self.b2_relu = KL.Activation('relu')

        self.c2 = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                                                     '2c', use_bias=use_bias)
        self.c2_batchnorm = BatchNorm(name=bn_name_base + '2c')

        self.id2 = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=use_bias)
        self.id2_batchnorm = BatchNorm(name=bn_name_base + '1')

        self.add = KL.Add()
        self.add_relu = KL.Activation('relu', name='res' + str(stage) + block + '_out')

    def call(self, input_tensor, train_bn=None):
        x = self.a2(input_tensor)
        x = self.a2_batchnorm(x, training=train_bn)
        x = self.a2_relu(x)

        x = self.b2(x)
        x = self.b2_batchnorm(x, training=train_bn)
        x = self.b2_relu(x)

        x = self.c2(x)
        x = self.c2_batchnorm(x, training=train_bn)

        shortcut = self.id2(input_tensor)
        shortcut = self.id2_batchnorm(shortcut, training=train_bn)

        x = self.add([x, shortcut])
        x = self.add_relu(x)
        return x

class Resnet(tf.keras.Model):

    def __init__(self, architecture='resnet50', **kwarg):
        super(Resnet, self).__init__(name=architecture, **kwarg)
        # Stage 1
        # self.zeropad = KL.ZeroPadding2D((3, 3))
        # self.conv1 = KL.Conv2D(64, (7, 4), strides=(2, 1), name='conv1', use_bias=True)
        # self.deform_conv1 = DeformableConvLayer(32, [5, 5], strides=(2, 1), num_deformable_group=1)  # out 24
        self.deform_conv = ConvOffset2D(32, name='conv12_offset')
        self.bn_conv = BatchNorm(name='bn_conv1')
        self.conv_relu = KL.Activation('relu')
        self.conv1 = KL.Conv2D(64, (3, 1), strides=(2, 1), name='conv1', use_bias=True)
        self.bn_conv1 = BatchNorm(name='bn_conv1')
        self.conv1_relu = KL.Activation('relu')
        self.C1 = KL.MaxPooling2D((3, 3), strides=(2, 1), padding="same")
        # Stage 2
        self.a2 = ConvBlock(3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        # self.b2 = IdentityBlock(3, [64, 64, 256], stage=2, block='b')
        self.C2  = IdentityBlock( 3, [64, 64, 256], stage=2, block='c')
        # Stage 3
        self.a3 = ConvBlock( 3, [128, 128, 512], stage=3, block='a')
        self.b3 = IdentityBlock( 3, [128, 128, 512], stage=3, block='b')
        # self.c3 = IdentityBlock( 3, [128, 128, 512], stage=3, block='c')
        self.C3 = IdentityBlock( 3, [128, 128, 512], stage=3, block='d')
        # Stage 4
        self.a4 = ConvBlock( 3, [256, 256, 512], stage=4, block='a')
        block_count = 4#{"resnet50": 5, "resnet101": 22}[architecture]
        self.identity_block4 = []
        for i in range(block_count-1):
            self.identity_block4.append(IdentityBlock( 3, [256, 256, 512], stage=4, block=chr(98 + i)))

        self.C4 = IdentityBlock( 3, [256, 256, 512], stage=4, block=chr(98 + block_count-1))
        # Stage 5
        # if tage5:
        self.a5 = ConvBlock( 3, [512, 512, 512], stage=5, block='a')
        # self.b5 = IdentityBlock( 3, [512, 512, 512], stage=5, block='b')
        self.C5 = IdentityBlock( 3, [512, 512, 512], stage=5, block='c')
        self.maxpool = KL.MaxPooling2D((3, 3), strides=(2, 1), padding="same")


    def call(self, input_tensor, training=True):
        # print(input_tensor.shape)
        # Stage 1
        # x = self.zeropad(input_tensor)
        # x = self.conv1(input_tensor)
        x = self.deform_conv(input_tensor)
        x = self.bn_conv(x, training=training)
        x = self.conv_relu(x)
        x = self.conv1(x)
        x = self.bn_conv1(x, training=training)
        x = self.conv1_relu(x)
        C1 = self.C1(x)
        # print(C1.shape)
        # Stage 2
        x = self.a2(C1, train_bn=training)
        # x = self.b2(x, train_bn=training)
        C2 = self.C2(x, train_bn=training)
        # print(C2.shape)
        # Stage 3
        x = self.a3(C2, train_bn=training)
        x = self.b3(x, train_bn=training)
        # x = self.c3(x, train_bn=training)
        C3 = self.C3(x, train_bn=training)
        # print(C3.shape)
        # Stage 4
        x = self.a4(C3, train_bn=training)
        for id_block4 in self.identity_block4:
            x = id_block4(x, train_bn=training)
        C4 = self.C4(x, train_bn=training)
        # print(C4.shape)
        # Stage 5
        # if stage5:
        x = self.a5(C4, train_bn=training)
        # x = self.b5(x, train_bn=training)
        C5 = self.C5(x, train_bn=training)
        C5 = self.maxpool(C5)
        # print(C5.shape)
        # else:
        #     C5 = None
        return C5

if __name__ == '__main__':
    model = Resnet('resnet50')
    inputs = tf.zeros((1, 224, 224, 3), tf.float32)
    for c in model(inputs):
        # pass
        print(c.shape)
    # num_variables = 0
    # for v in model.trainable_variables:
    #     size_var = tf.size(v).numpy()
    #     num_variables += size_var
    #     print(size_var)
    # print(num_variables)