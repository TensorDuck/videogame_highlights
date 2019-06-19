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

Developer Notes
===============
The .gitignore file ignores all files by default. If you want to add a
new file or filetype to the repo, the .gitignore file must be amended.

Credits
=======
This application uses Open Source components, specifically files contained in `loki/models/vggish_tensorflow/`. You can find the source code of their open source projects along with license information below. We acknowledge and are grateful to these developers for their contributions to open source.

Project: vggish https://github.com/tensorflow/models/tree/master/research/audioset/vggish

Copyright 2016 The TensorFlow Authors. All rights reserved.

License: Apache License 2.0 https://github.com/tensorflow/models/blob/master/LICENSE
