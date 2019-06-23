"""This package contains methods for using neural networks"""
import numpy as np
import os
import tensorflow as tf
from .vggish_tensorflow import CreateVGGishNetwork, EmbeddingsFromVGGish
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
    x_list -- list[numpy.ndarray]:
        List of traces of the sound wave (mono-channel)
    sr -- int:
        The sampling rate for the audio clip in Hz.
    """
    checkpoint_path = os.environ["SOUNDEMBEDDINGS"]

    print(checkpoint_path)
    all_embeddings = []
    tf.reset_default_graph()
    sess = tf.Session()

    vgg = CreateVGGishNetwork(sess, checkpoint_path)

    for x in x_list:
        resdict = EmbeddingsFromVGGish(sess, vgg, x, sr)
        all_embeddings.append(resdict['embedding'])

    sess.close()

    return all_embeddings

class SimpleNetwork(nn.Module):
    """A pytorch implementation of a final classification layer

    This is a simple model where a single linear unit is added after the
    embeddings layer from tensorflow. A sigmoid follows to infer the
    binary class.
    """
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.fc1 = nn.Linear(128,1)
        self.fc2 = nn.Sigmoid()

    def forward(self, x):
        y = self.fc1(x)
        y = self.fc2(y)
        return y

def stack_embeddings_and_targets(embeddings, targets=None):
    """Stack multiple embeddings along the zeroth axis

    Arguments:
    ----------
    embeddings -- list(np.array):
        A list of length L, with n_lx128 dimensional embeddings.

    Keyword Arguments:
    ------------------
    targets -- np.ndarray -- default=None:
        An array of target values. If given, will also stack and
        multiply the number of targets by n.

    Return:
    -------
    x -- np.ndarray:
        A Nx128 dimensional array where N = SUM_l(n_l)
    y -- np.ndarray:
        A N-length array.
    """
    x = np.zeros((0,128))
    y = []
    if targets is None:
        targets = np.zeros(len(embeddings))
    for tar,embed in zip(targets, embeddings):
        n_frames = np.shape(embed)[0]
        x = np.append(x, embed, axis=0)
        for i in range(n_frames):
            y.append(tar)

    return x,y

class NeuralNetworkClassifier():
    """Initialize a NN for learning on sound embeddings

    When training on clips longer than 1-second, their outputs are
    stacked such that you have an Nx128 dimensional array, where:
    N = SUM_i(clip_time_i)
    For the clip_time in seconds. It then trains 128-params to classify
    a scene as interesting or not interesting.

    Keyword Arguments:
    ------------------
    save_dir -- str -- default='./nn_model':
        Directory to save the model's learned parameters and log files.
    """

    def __init__(self):
        self.model = SimpleNetwork()

    def save(self, save_dir="./nn_model"):
        torch.save(self.model, save_dir)

    def load(self, target):
        self.model.load_state_dict(torch.load(target))

    def train(self, training_x, training_y, n_epochs=100, batch_size=None):
        """Train the neural network

        Arguments:
        ----------
        training_x -- list(np.ndarray):
            List of raw mono-audio traces sampled at 44.1kHz.
        training_y -- np.ndarray:
            Corresponding list of target classes for each audio pattern.

        Keyword Arguments:
        ------------------
        n_epochs -- int -- default=100:
            Number of training epochs to run.
        batch_size -- int -- default=all:
            Batch size of each training epoch. Default is all training
            data at each epoch.
        """
        x_train, y_train = stack_embeddings_and_targets(get_embeddings(training_x, 44100), targets=training_y)

        if batch_size is None: #set default batch-size
            batch_size = len(x_train)
        all_indices = np.arange(len(x_train)).astype(int)

        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        criterion = nn.MSELoss()
        for epoch in range(n_epochs):
            random_indices = np.random.choice(all_indices, size=batch_size, replace=False)
            total_loss = 0
            for i in random_indices:
                X = Variable(torch.FloatTensor([x_train[i]]), requires_grad=True)
                Y = Variable(torch.FloatTensor([y_train[i]]))
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, Y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch} Loss: {total_loss}")

    def infer(self, test_x):
        """Infer the classes on an inputted audio waveform. """
        embeddings_x = get_embeddings(test_x, 44100)
        inferred = []
        for x in embeddings_x:
            y = self.model(torch.FloatTensor(x))
            y_array = y.detach().numpy()
            avg = y_array.mean()
            if avg > 0.5:
                inferred.append(1)
            else:
                inferred.append(0)

        return np.array(inferred)
