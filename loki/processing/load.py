"""Class and methods for handling loading of video files"""
from moviepy.editor import VideoFileClip

class VideoClips():
    """Load multiple videos and write out relevant clips/audio

    VideoClips stores a series of video files and provides methods for
    editing and outputting clips from the video.

    Public Methods:
    ---------------
    save_clips():
        Save subclips of the loaded videofiles.
    """

    def __init__(self, filenames):
        """Load multiple videos and write out relevant clips/audio

        Uses moviepy for loading video frames and audio.

        Arguments:
        ----------
        filenames -- list(str):
            List of video filenames to load.
        """

        self.videos = []
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

        if write_names is None:
            write_names = []
            for stamp in time_stamps:
                write_names.append("vid%d_%d-%d.%s" % (stamp[0], stamp[1], stamp[2], write_ext))

        for i_count, stamp in enumerate(time_stamps):
            clip = self.videos[stamp[0]].subclip(stamp[1], stamp[2])
            clip.write_videofile(write_names[i_count], fps=write_fps)
