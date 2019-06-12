"""Class and methods for handling loading of video files"""
from moviepy.editor import VideoFileClip
from librosa import power_to_db
import numpy as np

class VideoClips():
    """Load multiple videos and write out relevant clips/audio

    VideoClips stores a series of video files and provides methods for
    editing and outputting clips from the video.

    Arguments:
    ----------
    filenames -- list(str):
        List of video filenames to load.

    Public Methods:
    ---------------
    save_clips():
        Save subclips of the loaded videofiles.
    """

    def __init__(self, filenames):
        self.videos = []
        self.audio_freq = None
        self.audios = None
        self.decibels = None
        for name in filenames:
            self.videos.append(VideoFileClip(name))

    def write_clips(self, time_stamps, write_fps=12, write_ext=".mp4", write_names=None):
        """Write selected clips to a file

        Save out N clips from the previously stored video clips.

        Arguments:
        ----------
        time_stamps -- Nx3 list([int, float, float]):
            Nx3 List giving the video index, followed by the start and
            stop times in seconds.

        Keyword Arguments:
        ------------------
        write_fps -- int -- default=12:
            Frames per a second to write out.
        write_ext -- str -- default="mp4":
            File extension format to save with.
        write_names -- list(str) -- default=None:
            List of len(N) to write output files to. If None, a default
            name format will be used.
        """

        #If write_names was not given, use a generic name output format
        if write_names is None:
            write_names = []
            for stamp in time_stamps:
                vid_idx = stamp[0]
                start_t = stamp[1]
                end_t = stamp[2]
                write_names.append(f"vid{vid_idx}_{start_t}-{end_t}{write_ext}")

        #Iterate over time_stamps and write out the specified clips
        for i_count, stamp in enumerate(time_stamps):
            clip = self.videos[stamp[0]].subclip(stamp[1], stamp[2])
            clip.write_videofile(write_names[i_count], fps=write_fps)

    def compute_audio_waveform(self, freq=44100):
        """Compute the binaural audio time series

        For each video stored, extract the binaural audio. This audio
        is then stored in the attribute self.audios, but also returns
        the list for use in further functions.

        Keyword Arguments:
        ------------------
        freq -- int -- default=44100:
            Frequency of the computed sound in Hz. Default is 44.1 kHz.

        Return:
        -------
        audios -- list(np.ndarray):
            Return a list of audio waveforms.
        """

        self.audio_freq = 44100
        self.audios = []
        for clip in self.videos:
            audio = clip.audio
            wav = audio.to_soundarray(fps=freq)
            self.audios.append(wav)

        return self.audios

    def compute_decibels(self):
        """Compute the total decibels from an audio waveform

        Compute the power by taking the square of the waveform. If the
        audio is binaural, then sum up the power of each audio channel.
        """

        self.decibels = []

        if self.audios is None:
            self.compute_audio_waveform()

        for binaural in self.audios:
            power = binaural ** 2 # square for the power
            #sum up binaural audio channel
            if power.ndim == 2:
                power = np.sum(power, axis=1)

            decibel = power_to_db(power)
            self.decibels.append(decibel)

        return self.decibels
