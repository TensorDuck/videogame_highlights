"""This package contains methods for using neural networks"""
import tensorflow as tf
from .vggish_tensorflow import CreateVGGishNetwork, EmbeddingsFromVGGish
import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

def get_embeddings(x_list, sr):
    """Get the sound embeddings from vgg-ish

    Use the pre-trained vggish network from TensorFlow in order to
    extract embeddings from an audio clip. The vggish network does all
    the preprocesisng necessary on a raw audio input.

    x can be arbitrary length, but the VGGish network was trained on
    0.96 second clips. As a result, the dimensions of the output is
    going to be 128 x M, where M = floor(time(x) / 0.96).
    i.e. a 10 second clip produces a 128 x 10 output.

    Arguments:
    ----------
    x -- numpy.ndarray:
        The trace of the sound wave (mono-channel)
    sr -- int:
        The sampling rate for the audio clip in Hz.
    """
    checkpoint_path = 'vggish_tensorflow/vggish_model.ckpt'

    tf.reset_default_graph()
    sess = tf.Session()


    vgg = CreateVGGishNetwork(sess, checkpoint_path)
    resdict = EmbeddingsFromVGGish(sess, vgg, x, sr)

    sess.close()

    return resdict['embedding']

class NeuralNetworkClassifier():
    """Initialize a NN for learning on sound embeddings

    Assume the embeddings use a 10x128 dimension for training and
    inference. This means that the input data is a 10-second long
    audio clip.

    Keyword Arguments:
    ------------------
    save_dir -- str -- default='./nn_model':
        Directory to save the model's learned parameters and log files.
    """

    def __init__(self, save_dir="./nn_model"):
        self.model = Sequential()
        self.model.add(Dense(1, input_shape=(10, 128), activation="sigmoid"))
        self.model.compile(optimizer='AdaDelta', loss='binary_crossentropy', metrics=['accuracy'])

        self.tb_callback = keras.callbacks.TensorBoard(log_dir=save_dir, histogram_freq=0, write_graph=True, write_images=False)

    def train(self, training_x, training_y, validation_x, validation_y):
        x_train = []
        x_valid = []

        for thing in training_x:
            x_train.append(get_embeddings(thing, 44100))
        for thing in training_y:
            x_valid.append(get_embeddings(thing,44100))
        self.model.fit(x_train, training_y, batch_size=10, epochs=200,
          verbose=0, validation_data=(x_valid, validation_y),callbacks=[tb_callback])
        score = model.evaluate(x_valid, validation_y, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


    def infer(self, test_x):
        pass
