"""Utility functions for the models sub-package"""

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
    scores -- list(float) or np.ndarray:
        The score of each scene, where higher numbers translate to
        more relevant scenes.
    clip_indices -- list([int, float, float]):
        List of video indices, and time stamps in seconds for each
        scene.
    """
    #argsort sorts lowest to highest so negate the score
    sort_indices = np.argsort(scores * -1)
    n_scenes = len(sort_indices)

    #use a while loop until n_top is found, hopefully this is short
    n_found = 0
    scene_index = 0
    best_scenes = []
    best_scores = []
    while n_found < n_top and scene_index < n_scenes:
        this_idx = sort_indices[scene_index]
        this_scene = clip_indices[this_idx]
        this_score = scores[this_idx]
        if check_overlap(best_scenes, this_scene):
            best_scenes.append(this_scene)
            best_scores.append(this_score)

        #increment scene_index by 1
        scene_index += 1

    return best_scores, best_scenes
