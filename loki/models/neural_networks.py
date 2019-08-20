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
    binary class. The embeddings from VGGish are a 128-Dimensional
    vector.
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
        A N-length array representing the stacked targets.
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
    y = np.array(y)

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
        """Save the pytorch model"""
        torch.save(self.model.state_dict(), save_dir)

    def load(self, target):
        """Load the pytorch model"""
        self.model.load_state_dict(torch.load(target))

    def train(self, training_x, training_y, n_epochs=100, batch_size=None, class_weights=None):
        """Train the neural network

        Training is done on a per-second basis, not on whole clips.
        Thus, clips are broken up into their constitutent seonds in this
        method. The batch_size is then effectively the number of seconds
        of audio data to trian on in each cycle. i.e. 10 clips of 10
        seconds each with a batch_size=20 means you train on 20% of the
        training_data in each epoch.

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
        class_weights -- np.ndarray -- default=None:
            Relative weight of each class. This weight affects the
            probability of picking each class when selecting the batch.
        """
        x_train, y_train = stack_embeddings_and_targets(get_embeddings(training_x, 44100), targets=training_y)

        if batch_size is None: #set default batch-size
            batch_size = len(x_train)
        if class_weights is None: #set default class_weights
            class_weights = np.ones(2)

        #pmatrix is the probability of selecting each class
        #pmatrix is based on the class_weights
        pmatrix = np.zeros(len(y_train))
        pmatrix[np.where(y_train == 0)] = class_weights[0]
        pmatrix[np.where(y_train == 1)] = class_weights[1]
        pmatrix /= np.sum(pmatrix)

        #all_indices is used for np.random.choice later
        all_indices = np.arange(len(x_train)).astype(int)

        #set pytroch optimizer and criterion
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        criterion = nn.MSELoss()
        #Begin the training epochs
        for epoch in range(n_epochs):
            #select random training indices for each batch
            random_indices = np.random.choice(all_indices, size=batch_size, replace=False, p=pmatrix)
            total_loss = 0
            #perform the pytorch training
            for i in random_indices:
                X = Variable(torch.FloatTensor([x_train[i]]), requires_grad=True)
                Y = Variable(torch.FloatTensor([y_train[i]]))
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, Y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            #print out the total loss every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch} Loss: {total_loss}")

    def infer(self, test_x, threshold=0.5):
        """Infer the classes on inputted audio waveform

        A clip is interesting if the average interest level over the
        whole clip is greater than a threshold of 0.5.

        Arguments:
        ----------
        test_x -- list[np.ndarray]:
            List of raw audio (mono-channel) waveforms.
        threshold -- float -- default=0.5:
            Threshold value for classifying into either class 1 or 0.
            If None, then return the raw non-thresholded scores.

        Return:
        -------
        inferred -- np.ndarray:
            Return the inferred classes.
        """
        embeddings_x = get_embeddings(test_x, 44100)
        inferred = []
        for x in embeddings_x:
            y = self.model(torch.FloatTensor(x))
            y_array = y.detach().numpy()
            avg = y_array.mean()
            if threshold is None:
                inferred.append(avg)
            else:
                if avg > threshold: #threshold is set to 0.5
                    inferred.append(1)
                else:
                    inferred.append(0)

        return np.array(inferred)

    def get_trace(self, test_x):
        """Get a trace of the interest level every 0.96 seconds

        Inputted audio waveforms are binned to every 0.96 seconds and
        then the interest level is inferred for each bin.

        Arguments:
        ----------
        test_x -- list[np.ndarray]:
            List of N raw audio (mono-channel) waveforms.

        Return:
        -------
        x_traces -- list[np.ndarray]:
            List of N arrays giving the time at the center of every
            0.96s long bin that the interest score was inferred over.
        traces -- list[np.ndarray]:
            List of N arrays giving the interest level of the
            corresponding time bin.
        """
        embeddings_x = get_embeddings(test_x, 44100)
        traces = []
        x_traces = []
        for x in embeddings_x:
            #perform the inference over the whole audio waveform at once
            y = self.model(torch.FloatTensor(x))
            y_array = y.detach().numpy()
            traces.append(y_array[:,0])

            #output the time stamps of each data point
            #time stamp is given at the center of each "bin"
            max_time = len(y_array) * 0.96
            time_stamps = np.arange(0.48, max_time, 0.96)
            x_traces.append(time_stamps)

        return x_traces, traces
