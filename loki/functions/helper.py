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
    print(fullpath_files)
    clips = processing.VideoClips(fullpath_files)

    return clips

def train_volume_classifier(train_clips, train_targets, test_clips=None, test_targets=None):
    """Get a trained volume classifier

    Return a trained volume classifier. If test_clips and test_targets
    is given, then also compute and print out validation statistics
    consisting of a confusion matrix, accuracy, precision and recall.

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

    if test_clips is not None and test_targets is not None:
        #compute validation of test_clips is given
        results = vclassifier.infer(processing.compute_decibels(test_clips))
        evaluation.print_confusion_matrix(test_targets, results)

    return vclassifier
