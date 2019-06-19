"""This package contains methods for using neural networks"""
import tensorflow as tf
from vggish_tensorflow import CreateVGGishNetwork, EmbeddingsFromVGGish
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense


def get_embeddings(x, sr):
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
    def __init__():
        pass

    def train(self, training_x, training_y):
        pass

    def infer(self, test_x):
        pass
