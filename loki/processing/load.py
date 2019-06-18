"""Class and methods for handling loading of video files"""
from moviepy.editor import VideoFileClip

def append_clips(first, second):
    """Append two different VideoClips objects

    Arguments:
    ----------
    first -- loki.VideoClips:
        These filenames will go first.
    second -- loki.VideoClips:
        These filenames will follow the filenames in first.

    Return:
    -------
    vclips -- loki.VideoClips:
        A new VideoClips object with both sets of filenames stored.
    """
    #collect the filenames
    all_filenames = []
    for fil in first.filenames:
        all_filenames.append(fil)
    for fil in second.filenames:
        all_filenames.append(fil)

    #make the new VideoClips, does not support saving audio information
    vclips = VideoClips(all_filenames)

    return vclips

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
        #save the filenames and avoid pass by reference errors
        self.filenames = filenames[:]

        #these are attributes resultant from later analysis
        self.audio_freq = None
        self.audios = None

    @property
    def nclips(self):
        return len(self.filenames)

    def write_clips(self, time_stamps, write_fps=12, write_ext=".mp4", write_names=None):
        """Write selected clips to a file

        Save out N clips from the previously stored video clips.

        Arguments:
        ----------
        time_stamps -- Nx3 list or np.ndarray:
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
                vid_idx = int(stamp[0])
                start_t = stamp[1]
                end_t = stamp[2]
                write_names.append(f"vid{vid_idx}_{start_t}-{end_t}{write_ext}")

        #Iterate over time_stamps and write out the specified clips
        for i_count, stamp in enumerate(time_stamps):
            this_vid = VideoFileClip(self.filenames[0])
            clip = this_vid.subclip(stamp[1], stamp[2])
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
        for fname in self.filenames:
            clip = VideoFileClip(fname)
            audio = clip.audio
            wav = audio.to_soundarray(fps=freq)
            self.audios.append(wav)

        return self.audios
