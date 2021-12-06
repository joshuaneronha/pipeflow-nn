import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Conv2DTranspose
# import tensorflow_probability as tfp

class CNNAutoEncoder(tf.keras.Model):

    def __init__(self):
        super(CNNAutoEncoder, self).__init__()

        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)  #.01 originally, improved w/ .005
        self.batch_size = 32
        self.epochs = 25  #originally 25 dec5, increasing to 35 doesn't help at all

        self.encoder_1 = Conv2D(128, 4, 4, 'same',activation='relu')

        self.encoder_2 = tf.keras.Sequential()
        self.encoder_2.add(Conv2D(256, 4, 5, 'same',activation='relu'))
        self.encoder_2.add(Conv2D(512, 4, 2, 'same',activation='relu'))

        self.dense = tf.keras.layers.Dense(1024, activation = 'relu')

        self.decoder_ux_1 = tf.keras.Sequential()
        self.decoder_ux_1.add(Conv2DTranspose(256, 4, 2, 'same',activation='relu'))
        self.decoder_ux_1.add(Conv2DTranspose(128, 4, 5, 'same',activation='relu'))

        self.decoder_ux_2 = tf.keras.Sequential()
        self.decoder_ux_2.add(Conv2DTranspose(64, 4, 2, 'same',activation='relu'))
        self.decoder_ux_2.add(Conv2DTranspose(1, 4, 2, 'same'))



        self.decoder_uy_1 = tf.keras.Sequential()
        self.decoder_uy_1.add(Conv2DTranspose(256, 4, 2, 'same',activation='relu'))
        self.decoder_uy_1.add(Conv2DTranspose(128, 4, 5, 'same',activation='relu'))

        self.decoder_uy_2 = tf.keras.Sequential()
        self.decoder_uy_2.add(Conv2DTranspose(64, 4, 2, 'same',activation='relu'))
        self.decoder_uy_2.add(Conv2DTranspose(1, 4, 2, 'same'))



        self.decoder_p_1 = tf.keras.Sequential()
        self.decoder_p_1.add(Conv2DTranspose(256, 4, 2, 'same',activation='relu'))
        self.decoder_p_1.add(Conv2DTranspose(128, 4, 5, 'same',activation='relu'))

        self.decoder_p_2 = tf.keras.Sequential()
        self.decoder_p_2.add(Conv2DTranspose(64, 4, 2, 'same',activation='relu'))
        self.decoder_p_2.add(Conv2DTranspose(1, 4, 2, 'same'))
    def call(self, input):

        residual = self.encoder_1(input)
        encoded = self.encoder_2(residual)
        densified = self.dense(encoded)

        ux_residual = self.decoder_ux_1(densified) + residual
        uy_residual = self.decoder_uy_1(densified) + residual
        p_residual = self.decoder_p_1(densified) + residual

        ux = self.decoder_ux_2(ux_residual) * tf.cast(input,tf.float32)
        uy = self.decoder_uy_2(uy_residual) * tf.cast(input,tf.float32)
        p = self.decoder_p_2(p_residual) * tf.cast(input,tf.float32)

        return ux, uy, p

    def loss_function(self,prediction,true,mask):
        try:
            prediction = tf.squeeze(prediction,axis=3)
        except:
            pass
        try:
            mask = tf.cast(tf.squeeze(mask,axis=3),tf.float32)
        except:
            pass
        # rms_loss = tf.sqrt(tf.cast(tf.reduce_sum(tf.square((prediction - true) * tf.cast(mask,tf.float32)),axis=[1,2]),tf.float32) / tf.cast(tf.reduce_sum(mask,axis=[1,2]),tf.float32))
        # avgd = tf.reduce_mean(rms_loss)
        # RMS = tf.keras.metrics.RootMeanSquaredError()
        # RMS.update_state(true, prediction, sample_weight = mask)
        # print(tf.convert_to_tensor(RMS.result().numpy()))
        # return tf.convert_to_tensor(RMS.result().numpy())
        # mse = tf.keras.losses.MeanSquaredError(reduction = ttf.keras.losses.Reduction.NONE)
        # print(mse(true,prediction).shape)

        a = tf.reduce_sum(tf.square(prediction - true) * mask,axis=[1,2]) / tf.reduce_sum(mask)
        return tf.reduce_mean(a)

    def prediction_error(self,prediction,true,mask):
        try:
            prediction = tf.squeeze(prediction,axis=3)
        except:
            pass
        try:
            mask = tf.cast(tf.squeeze(mask,axis=3),tf.float32)
        except:
            pass


        numDataPoints = tf.reduce_sum(mask)
        rawErrors = (tf.math.abs(prediction - true) * mask)
        avgRawError = tf.reduce_sum(rawErrors)/ numDataPoints

        # after removing 'masked' values, can calculate medians:
        maskBoolean = tf.cast(mask, tf.bool)
        maskedPrediction = prediction[maskBoolean]
        maskedTruth = true[maskBoolean]
        shortenedRawErrors = tf.math.abs(maskedPrediction - maskedTruth)
        shortenedPercErrors = tf.math.abs(shortenedRawErrors/maskedTruth) * 100
        medianPercError = np.median(shortenedPercErrors)

        avgValueMagnitude = tf.reduce_sum(tf.math.abs(true)*mask) / numDataPoints
        normalizedAvgError = avgRawError/avgValueMagnitude * 100 #kind of like a percentage error

        return avgRawError, medianPercError, normalizedAvgError