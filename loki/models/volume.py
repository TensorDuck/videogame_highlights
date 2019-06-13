"""Class and methods for detecting loudest portions of videos"""
import math
import numpy as np

from .util import sort_scores_and_remove_overlap

class VolumeModel():
    """Find the loudest sections in a set of videos

    Keyword Arguments:
    ------------------
    search_length -- float -- default=10.0:
        Desired clip size in seconds.

    search_increment -- float -- default=10.0:
        Desired shift to apply to search window in seconds.

    """

    def __init__(self, search_length=10.0, search_increment=1.0):
        self.search_length = search_length
        self.search_increment = search_increment

    def predict(self, loudness, freq=44100, n_predict=1):
        """Find the loudest section in the inputted video clips

        Take the input loudness generated from a video and search over
        every volume array to determine the clips with the overall
        loudest moments. Return the video index and time index
        corresponding to the overall loudest portion.

        Arguments:
        ----------
        loudness -- list[np.ndarray(float)]:
            List of all total volume of multiple vidoeo clips.

        Keywrod Arguments:
        ------------------
        freq -- int -- default=44100
            Frequency in Hz to extract the audio over.
        n_predict -- int -- default=1
            Return the top n_predict non-overlapping scenes.
        """

        #Define search windows in array index lengths
        search_window = math.floor(self.search_length * freq)
        search_jump = math.floor(self.search_increment * freq)

        #store the loudest section and increment
        all_loudness_scores = np.zeros(0)
        all_scenes = np.zeros((0,3))

        #check each audio clip
        for audio_idx, audioclip in enumerate(loudness):
            #check if clip is longer than search window
            if len(audioclip) <= search_window:
                #if longer, compare average volume of this portion
                avg_loudness = np.sum(audioclip) / float(len(audioclip))
                clip_increment = np.array([audio_idx, 0, len(audioclip)/freq]).reshape((1,3))
                #append to lists
                all_loudness_scores = np.append(all_loudness_scores, avg_loudness)
                all_scenes = np.append(all_scenes, clip_increment, axis=0)
            else:
                #If clip is not longer, check every window
                start_indices = range(0, len(audioclip) - search_window, search_jump)
                #Increment over every window
                for start_idx in start_indices:
                    end_idx = start_idx + search_window
                    avg_loudness = np.sum(audioclip[start_idx:end_idx]) / float(search_window)
                    clip_increment = np.array([audio_idx, start_idx/freq, end_idx/freq]).reshape((1,3))
                    #append to lists
                    all_loudness_scores = np.append(all_loudness_scores, avg_loudness)
                    all_scenes = np.append(all_scenes, clip_increment, axis=0)

        #return the top scores
        top_scores, top_scenes = sort_scores_and_remove_overlap(n_predict, all_loudness_scores, all_scenes)

        return top_scores, top_scenes
