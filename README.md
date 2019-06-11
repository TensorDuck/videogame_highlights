# videogame_highlights
Machine Learning to automatically generate highlights from videogame streams

Installation
============

To add the relevant directories to the PYTHONPATH variable:

```
cd build
source add_path.sh
```

The methods in the LOKI analysis package should then be usable as:

```
import loki

clips = loki.VideoClips(["example.mp4"])
```
