"""Utility functions for the models sub-package"""
import numpy as np

def sort_scores_and_remove_overlap(n_top, scores, clip_indices):
    """Sort based on the inputted scores and return the n_top scores.

    Overlap is determined where the scene with the highest score is
    kept. Subsequent scenes with overlapping time indices are then
    ignored. This process is repeated until n_top non-overlapping scenes
    are found.

    Scores can be any value that characterizes the interest level of a
    scene, with the assumption that higher scores = higher interest.
    For example, this could be the average volume of a scene or some
    inferred interest level from some classifier.

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

    #use a while loop until n_top are found, hopefully this is short
    n_found = 0 #count number of non-overlapping scores found
    scene_index = 0 #keep track of number of scenes
    best_scenes = np.zeros((n_top,3))
    best_scores = []
    while n_found < n_top and scene_index < n_scenes:
        #terminate the while loop if every scene is checked.
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
    scenes. The Format of each check_scene and elements in all_scenes is
    the same. The first element is an integer that denotes the video
    index the scene is from. The next two elements are floats that
    denote the start and stop times respectively. Therefore, check_scene
    is not overlapping if its from a different video than a scene in
    all_scenes. If they are from the same video, then they are not
    overlapping if check_scene finishes before or starts after the
    other scene.

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
            #first two check if the `scene` happens after `check_scene`
            #last two checks if the `scene` happens before `check_scene`
            #If all the checks are true, then keep going
            #If one of the checks fail, break from loop and return True
            if not (scene[1] > check_scene[1] and scene[1] > check_scene[2] and scene[2] < check_scene[1] and scene[2] < check_scene[2]):
                return True

    #If the function gets here, there is no overlap
    return False
