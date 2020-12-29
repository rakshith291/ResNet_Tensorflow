import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

class IdentityBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size,one_cross=False):
        super(IdentityBlock, self).__init__(name='')

        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.act = tf.keras.layers.Activation('relu')
        self.add = tf.keras.layers.Add()

        self.one_cross = one_cross
        self.conv = tf.keras.layers.Conv2D(filters,1)

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        if self.one_cross == False:
          x = self.add([x, input_tensor])
        else :
          input_tensor = self.conv(input_tensor)
          x = self.add([x, input_tensor])
        x = self.act(x)
        return x

class ResNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=(3, 3))
        self.conv = tf.keras.layers.Conv2D(64, 7, strides=(2,2))
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
        self.pad2 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
        self.max_pool = tf.keras.layers.MaxPool2D(3,strides=(2, 2))

        # Use the Identity blocks that you just defined
        self.id1a = IdentityBlock(64, 3)
        self.id1b = IdentityBlock(64, 3)

        self.id2a = IdentityBlock(128,3,True)
        self.id2b = IdentityBlock(128,3,True)

        self.id3a = IdentityBlock(256,3,True)
        self.id3b = IdentityBlock(256,3,True)

        self.id4a = IdentityBlock(512,3,True)
        self.id4b = IdentityBlock(512,3,True)

        self.avg_pool = tf.keras.layers.AveragePooling2D((7,7))
        self.dense = tf.keras.layers.Dense(1000)
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.pad1(inputs)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        x = self.pad2(x)
        x = self.max_pool(x)
        x = self.id1a(x)
        x = self.id1b(x)

        x = self.pad2(x)
        x = self.max_pool(x)
        x = self.id2a(x)
        x = self.id2b(x)

        x = self.pad2(x)
        x = self.max_pool(x)
        x = self.id3a(x)
        x = self.id3b(x)

        x = self.pad2(x)
        x = self.max_pool(x)
        x = self.id4a(x)
        x = self.id4b(x)

        x = self.avg_pool(x)
        x = self.dense(x)
        x = self.act(x)

        return self.classifier(x)

    def model(self):
        x = Input(shape=(224, 224, 3))
        return Model(inputs=[x], outputs=self.call(x))