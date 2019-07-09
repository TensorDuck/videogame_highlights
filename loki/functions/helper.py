"""Contains helper functions for loading/training/evaluation"""
import os
import numpy as np

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

def average_over_window(data, n_average):
    """Computer a sliding window average over data

    Given a window size in indices, compute the average value of data
    over that window sliding by one index.

    Arguments:
    ----------
    data -- np.ndarray:
        1-D array of length N to compute averages over.
    n_average -- int:
        Size of the window.

    Return:
    -------
    new_data -- np.ndarray:
        1-D array of length N-n_average. Each index represents the
        average over n_average consecutive elements.

    Example:
    --------
    Given data = [0, 1, 2, 3, 4]
    n_average = 2
    Then the average trace over a window of 2 is:
    [0.5, 1.5, 2.5, 3.5]
    """

    new_data = np.copy(data)[:-n_average]
    for i in range(1, n_average):
        new_data += data[i:-(n_average-i)]
    new_data /= n_average

    return new_data

def find_best_clip(video_files, clip_length, nn_checkpoint="nn_model"):
    """Find the best clip in a set of videos of specified duration

    Search over every video_file and compute a windowed average of the
    interest level every second. The clip section with the largest
    average interest is then returned, as well as the original
    non-averaged trace of interest level for each inputted video file.

    Arguments:
    ----------
    video_files -- list[str]:
        List of N video files to calculcate interest levels for.
    clip_length -- float:
        Length of the desired highlight clip in seconds.

    Keyword Arguments:
    ------------------
    nn_model -- str -- nn_model:
        Location of the loki.NeuralNetworkClassifier checkpoint file.

    Return:
    -------
    best_clip -- list:

    -- dict:
        Return a dictionary containing the best_clip, x_trace, and
        y_trace.
        best_clip -- list:
            Contains the best clip section. The first element is the
            video file containing the best clip. The second and third
            element is the start and stop time respectively.
        x_trace -- list[np.ndarray]:
            List of N arrays giving the times for the center of each
            averaging window.
        y_trace -- list[np.ndarray]:
            List of N arrays giving the average interst level over each
            window.
    """
    #0.96 is the length of time VGGish processes as a single embedding
    clip_size = int(np.ceil(clip_length / 0.96))
    nnclass = models.NeuralNetworkClassifier()
    nnclass.load(nn_checkpoint)
    vclips = processing.VideoClips(video_files)
    big_audio = vclips.compute_audio_waveform()

    x_trace, y_trace = nnclass.get_trace(big_audio)

    #save the average interest level over each clip in a windowed avg
    x_avg = []
    y_avg = []
    for x,y in zip(x_trace,y_trace):
        x_avg.append(average_over_window(x, clip_size))
        y_avg.append(average_over_window(y, clip_size))

    #find the most interesting segment
    best_interest = 0
    best_time = 0
    best_file = None
    for i,(x,y) in enumerate(zip(x_avg, y_avg)):
        if np.max(y) > best_interest:
            #found a better clip
            best_interest = np.max(y)
            #find time of peak interest. If multiple, keep only first
            best_time = x[np.where(y == best_interest)][0]
            best_file = video_files[i]

    half_clip = clip_length * 0.5
    best_clip = [best_file, best_time - half_clip, best_time +half_clip]

    return {"best_clip":best_clip, "x_trace":x_trace, "y_trace":y_trace}

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
        clf = _train_volume_classifier(train_clips, train_targets)
    elif classifier == "nn":
        clf = _train_nn_classifier(train_clips, train_targets, n_epochs=n_epochs, class_weights=class_weights, batch_size=batch_size)
    else:
        print("Invalid Classifier Specified. Keyword wargument classifier must be either 'volume' or 'nn'.")

    if test_clips is not None and test_targets is not None:
        #compute validation of test_clips is given
        if classifier == "volume":
            results = clf.infer(processing.compute_decibels(test_clips))
        elif classifier == "nn":
            results = clf.infer(test_clips.compute_audio_waveform(mono=True))
        else:
            print("Invalid Classifier Specified. Keyword wargument classifier must be either 'volume' or 'nn'.")
        evaluation.print_confusion_matrix(test_targets, results)

    return clf

def _train_volume_classifier(train_clips, train_targets):
    """Get a trained volume classifier

    Arguments:
    ----------
    See loki.functions.helper.train_classifier().

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
    nclassifier.train(raw_audio, train_targets, n_epochs=n_epochs, class_weights=class_weights, batch_size=batch_size)

    return nclassifier
