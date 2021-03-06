"""Class and methods for detecting loudest portions of videos"""
import math
import numpy as np
import sklearn.metrics as skmet

from .util import sort_scores_and_remove_overlap

class VolumeClassifier():
    """A classifier that classifiers interesting scenes based on volume

    This classifier will class a scene as interesting if its average
    volume is above a certain cutoff. Values above that cutoff are then
    classified as interesting.

    Attributes:
    -----------
    volume_cutoff -- float:
        Scenes with average volume above volume_cutoff are classified as
        interesting. Those less than or equal to volume_cutoff are
        classified as uninteresting.
    """

    def __init__(self):
        self.volume_cutoff = 0

    def save(self, savefile=None):
        """Save the model's learned parameters

        Write out the volume_cutoff to a file. If no file is specified,
        then write into volume_classifier_model.dat in the current
        directory.

        Keyword Arguments:
        ------------------
        savefile -- str -- default=None:
            The file name to write the parameters into.
        """
        if savefile is None:
            savefile = "volume_classifier_model.dat"
        f = open(savefile, "w")
        f.write(f"volume_cutoff = {self.volume_cutoff}\n")
        f.close()

    def load(self, savefile):
        """Load a previous model's learned parameters

        Arguments:
        ----------
        savefile -- str -- default=None:
            The file name to write the parameters into.
        """
        f = open(savefile, "r")

        for line in f:
            #load the parameters one by one
            stuff = line.strip().split()
            if stuff[0] == "volume_cutoff":
                self.volume_cutoff = float(stuff[2])
        f.close()


    def train(self, training_x, training_y):
        """Train the volume classifier

        Train the volume classifier (currently a binary classifier).
        Default loss function is the hamming loss which, for a binary
        classifier, is equivalent to accuracy.
        The training here is simply finding the threshold that maximizes
        the accuracy. It's not technically training but the method is
        named such for consistency.

        Arguments
        ---------
        training_x -- list[np.ndarray]:
            The volume (in decibels) of the training data. Of length N.
        training_y -- np.ndarray or list:
            The corresponding classes of the training data. Also of
            length N. A class of 1 is interesting, a class of 0 is
            uninteresting.
        """
        average_loudness = []
        for audioclip in training_x:
            average_loudness.append(audioclip.mean())
        average_loudness = np.array(average_loudness)

        best_loss = 1 # hamming loss goes from 0 - 1
        best_cutoff = None

        #Get a unique set of volume cutoffs:
        unique_values = np.unique(average_loudness)
        possible_cutoffs = unique_values[1:] + unique_values[:-1]
        low_endvalue = np.min(unique_values) - possible_cutoffs[1] + possible_cutoffs[2]
        high_endvalue = np.max(unique_values)

        possible_cutoffs = np.append([low_endvalue, high_endvalue], possible_cutoffs)

        #check every possible cutoff
        for cutoff in possible_cutoffs:
            predicted_values = np.zeros(len(average_loudness))
            predicted_values[np.where(average_loudness > cutoff)] = 1
            loss = skmet.hamming_loss(training_y, predicted_values)
            if loss < best_loss:
                best_loss = loss
                best_cutoff = cutoff

        self.volume_cutoff = best_cutoff

    def infer(self, test_x, score=False):
        """Make an inference on the test data based on trained model

        For instances in test_x where the average volume is greater than
        self.volume_cutoff, class it as 1 for interesting.

        Arguments:
        ----------
        test_x -- list[np.ndarray]:
            The volume (in decibels) of the test data.
        score -- bool -- default=False:
            If False, return the classes based on the volume threshold.
            If True, return the non-thresholded average volume of each
            scene.

        Return:
        -------
        classified -- np.ndarray:
            The resultant classes for the test data.
         """
        classified = []
        for audioclip in test_x:
            avg_volume = audioclip.mean()
            if score:
                classified.append(avg_volume)
            else:
                if avg_volume > self.volume_cutoff:
                    classified.append(1)
                else:
                    classified.append(0)

        return np.array(classified)


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

        Note, the inputted loudness are not expected to be the same
        length as the length of videos would vary a great deal. Assuming
        every video is atleast longer than self.search_length, then the
        outputted loudest segments would be of standard length.

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
