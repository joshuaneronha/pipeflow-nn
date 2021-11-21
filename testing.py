import tensorflow as tf
from skimage import transform
from preprocess import import_data
import matplotlib.pyplot as plt

help = tf.random.uniform((10,80,160,1))

plt.imshow(help[0])

# squeezed = tf.squeeze(help)
#
# tf.sqrt(tf.cast(tf.reduce_sum(tf.square((tf.squeeze(squeezed) - squeezed) * tf.cast(tf.squeeze(squeezed),tf.float32)),axis=[1,2]),tf.float32) / tf.cast(tf.reduce_sum(squeezed,axis=[1,2]),tf.float32)).shape
#
# conv1 = tf.keras.layers.Conv2D(128,4,strides=4,padding='same')
# conv2 = tf.keras.layers.Conv2D(256,4,strides=2,padding='same')
# conv3 = tf.keras.layers.Conv2D(512,4,strides=2,padding='same')
# tconv1 = tf.keras.layers.Conv2DTranspose(128,4,strides=4,padding='same')
# tconv2 = tf.keras.layers.Conv2DTranspose(128,4,strides=2,padding='same')
# tconv3 = tf.keras.layers.Conv2DTranspose(1,4,strides=2,padding='same')
# dense = tf.keras.layers.Dense(1024)
#
# help.shape
# a = conv1(help)
# a.shape
# b = conv2(a)
# b.shape
# c = conv3(b)
# c.shape
# d = dense(c)
# d.shape
# e = tconv1(d)
# e.shape
# f = tconv2(e)
# f.shape
# g = tconv3(f)
# g.shape
#
# from sklearn.model_selection import train_test_split
#
# g, r = import_data()
#
# r.shape
# train_test_split(g,r)[1].shape
