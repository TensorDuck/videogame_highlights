"""Contains helper functions for loading/training/evaluation"""
import os
from .. import processing
from .. import models
from . import evaluation

def load_clips_from_dir(target_dir=None):
    """Make a VideoClips object with all files in a dir

    Keyword Arguments:
    ------------------
    target_dir -- str -- None:
        The target dir to load the files from.

    Return:
    -------
    clips -- loki.VideoClips:
        A VideoClips loader with all the files in target_dir.
    """
    cwd = os.getcwd()
    if target_dir is None:
        #default load from current directory
        target_dir = cwd
    else:
        #make sure you use the full path
        os.chdir(target_dir)
        target_dir = os.getcwd()
        os.chdir(cwd)

    #grab every file and alphabetize
    all_files = os.listdir(target_dir)
    all_files.sort()

    #append the fullpath to every file
    fullpath_files = []
    for fil in all_files:
        fullpath_files.append(f"{target_dir}/{fil}")

    clips = processing.VideoClips(fullpath_files)

    return clips

def train_classifier(train_clips, train_targets, test_clips=None, test_targets=None, classifier="nn", n_epochs=100, batch_size=None, class_weights=None):
    """Get a trained classifier for audio data

    Return a trained classifier. If test_clips and test_targets is
    given, then also compute and print out validation statistics
    consisting of a confusion matrix, accuracy, precision and recall.
    Default is to train a NeuralNetworkClassifier.

    Arguments:
    ----------
    train_clips -- loki.VideoClips:
        The loaded video clips to use for training.
    train_targets -- np.ndarray:
        An array with the same number of elements as train_clips
        classifying each clip as either interesting (1) or boring (0).

    Keyword Arguments:
    ------------------
    test_clips -- loki.VideoClips -- default=None:
        The loaded video clips used for validation.
    test_targets -- np.ndarray -- default=None:
        An array with the same number of elements as test_clips
        classifying each clip as either interesting (1) or boring(0)
    classifier -- str -- default='nn':
        Type of classifier to train. `nn` returns a
        loki.NeuralNetworkClassifier while 'volume' returns a
        loki.VolumeClassifier.
    n_epochs -- int -- default=100:
        Number of training epochs to run. Only used for classifier=`nn`.
    batch_size -- int -- default=all:
        Batch size of each training epoch. Default is all training
        data at each epoch. Only used for classifier=`nn`.
    class_weights -- np.ndarray -- default=None:
        Relative weight of each class. This weight affects the
        probability of picking each class when selecting the batch. Only
        used for classifier=`nn`.

    Return:
    -------
    classifier -- loki.VolumeClassifier:
        A trained classifier.
    """
    if classifier == "volume":
        classifer = _train_volume_classifier(train_clips, train_targets)
    elif classifier == "nn":
        classifier = _train_nn_classifier(train_clips, train_targets)
    else:
        print("Invalid Classifier Specified. Keyword wargument classifier must be either 'volume' or 'nn'.")

    if test_clips is not None and test_targets is not None:
        #compute validation of test_clips is given
        results = classifier.infer(processing.compute_decibels(test_clips))
        evaluation.print_confusion_matrix(test_targets, results)

    return classifier

def _train_volume_classifier(train_clips, train_targets):
    """Get a trained volume classifier

    Arguments:
    ----------
    train_clips -- loki.VideoClips:
        The loaded video clips to use for training.
    train_targets -- np.ndarray:
        An array with the same number of elements as train_clips
        classifying each clip as either interesting (1) or boring (0).

    Return:
    -------
    vclassifier -- loki.VolumeClassifier:
        A trained volume classifier.

    """
    vclassifier = models.VolumeClassifier()

    #extract the audio data
    audio_data = processing.compute_decibels(train_clips)
    #train the volume classifier
    vclassifier.train(audio_data, train_targets)

    return vclassifier

def _train_nn_classifier(train_clips, train_targets, n_epochs=100, batch_size=None, class_weights=None):
    """Get a trained neural network classifier

    Arguments:
    ----------
    See loki.functions.helper.train_classifier().

    Return:
    -------
    nclassifier -- loki.NeuralNetworkClassifier:
        A trained neural network classifier.
    """
    nclassifier = models.NeuralNetworkClassifier()

    raw_audio = train_clips.compute_audio_waveform(mono=True)
    nclassifier.train(raw_audio, train_targets, n_epochs="n_epochs", class_weights="class_weights", batch_size="batch_size")

    return nclassifier
