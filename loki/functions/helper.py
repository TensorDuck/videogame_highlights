"""Contains helper functions for loading/training/evaluation"""
import os
from .. import processing
from .. import models

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
