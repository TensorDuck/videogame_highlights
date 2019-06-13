"""Utility functions for the models sub-package"""
import numpy as np

def sort_scores_and_remove_overlap(n_top, scores, clip_indices):
    """Sort the scores and return the n_top scores.

    Overlap is determined where the scene with the highest score is
    kept. Subsequent scenes with overlapping time indices are then
    ignored. This process is repeated until n_top non-overlapping scenes
    are found.

    Arguments:
    ----------
    n_top -- int:
        The number of top scoring scenes to return.
    scores -- np.ndarray(N,):
        The score for each of the N scenes, where higher numbers
        translate to more relevant scenes.
    clip_indices -- list([int, float, float]):
        List of video indices, and time stamps in seconds for each
        scene.

    Return:
    -------
    best_scores -- np.ndarray(float):
        The score for the corresponding scene.
    best_scenes -- np.ndarray(float(n_top,3)):
        The highest scoring scenes formatted as
        [video index, start time, stop time]
    """
    #argsort sorts lowest to highest so negate the score
    sort_indices = np.argsort(scores * -1)
    n_scenes = len(sort_indices)

    #use a while loop until n_top is found, hopefully this is short
    n_found = 0
    scene_index = 0
    best_scenes = np.zeros((n_top,3))
    best_scores = []
    while n_found < n_top and scene_index < n_scenes:
        this_idx = sort_indices[scene_index]
        this_scene = clip_indices[this_idx]
        this_score = scores[this_idx]
        #check if overlapping
        if not is_overlapping(best_scenes, this_scene):
            best_scenes[n_found,:] = this_scene
            best_scores.append(this_score)
            #increment found index by 1
            n_found += 1

        #increment scene_index by 1
        scene_index += 1

    return best_scores, best_scenes

def is_overlapping(all_scenes, check_scene):
    """Check the check_scene against all_scenes for overlap

    check_overlap() returns True if there is any overlap with previous
    scenes

    Arguments:
    ----------
    all_scenes -- list(list([int, float, float])):
        List of all scenes to check against.
    check_scene -- list([int, float, float]):
        The scene you want to check for.

    Return:
    -------
    bool
    """
    for scene in all_scenes:
        #check if it's the same video
        if scene[0] == check_scene[0]:
            #check if there's any overlap
            if scene[1] > check_scene[1] and scene[1] > check_scene[2]:
                #scene happens after check_scene
                pass
            elif scene[2] < check_scene[1] and scene[2] < check_scene[2]:
                #scene happens before check_scene
                pass
            else:
                #there is overlap here
                return True

    #If the function gets here, there is no overlap
    return False
