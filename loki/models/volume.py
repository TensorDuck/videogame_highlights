"""Class and methods for detecting loudest portions of videos"""
import math
import numpy as np

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

    def predict(self, vclips, freq=44100):
        """Find the loudest section in the inputted video clips

        Take the input vclips and search over every video. Compute the
        total volume in the increment self.search_length and then
        return the video index and time index corresponding to the
        overall loudest portion.

        Arguments:
        ----------
        vclips -- loki.VideoClips:
            The VideoClips object containing the desired video clips to
            search over.

        Keywrod Arguments:
        ------------------
        freq -- int -- default=44100
            Frequency in Hz to extract the audio over.
        """

        #extract the volume from vclips
        loudness = vclips.compute_decibels(freq=freq)

        #Define search windows in array index lengths
        search_window = math.floor(self.search_length * freq)
        search_jump = math.floor(self.search_increment * freq)

        #store the loudest section and increment
        loudest = None
        loudest_increment = None

        #check each audio clip
        for audio_idx, audioclip in enumerate(loudness):
            #check if clip is longer than search window
            if len(audioclip) <= search_window:
                #if longer, compare average volume of this portion
                avg_loudness = np.sum(audioclip) / float(len(audioclip))
                clip_increment = [audio_idx, 0, len(audioclip)/freq]
                #check with previous results
                if loudness is None or avg_loudness > loudest:
                    loudest = avg_loudness
                    loudest_increment = clip_increment
            else:
                #If clip is not longer, check every window
                start_indices = range(0, len(audio_idx) - search_window, search_jump)
                #Increment over every window
                for start_idx in start_indices:
                    end_idx = start_idx + search_window
                    avg_loudness = np.sum(audioclip[start_idx:end_idx]) / float(search_window)
                    clip_increment = [audio_idx, start_idx/freq, end_idx/freq]
                    #check with previous results
                    if loudness is None or avg_loudness > loudest:
                        loudest = avg_loudness
                        loudest_increment = clip_increment

        return loudest_increment, avg_loudness
