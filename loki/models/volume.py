"""Class and methods for detecting loudest portions of videos"""

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
