# videogame_highlights
Machine Learning to automatically generate highlights from videogame streams

Getting Started
===============

Prerequisites
-------------
This package was written and tested for `Python 3.7.3` compiled using `GCC 7.3.0` in a `conda` environment.
The following packages (and their dependencies) would need to be installed.
The versions listed have been tested in my environment and works, but likely any more recent or backwards compatible version of those packages would also work.

- [`numpy=1.16.3`](http://www.numpy.org/)
- [`scipy=1.2.1`](http://www.scipy.org/)
- [`tensorflow=1.13.1`](http://www.tensorflow.org/)
- [`pytorch-cpu=1.1.0`](https://pytorch.org/)
- [`scikit-learn=0.21.1`](https://scikit-learn.org/)
- [`moviepy==1.0.0`](https://zulko.github.io/moviepy/)
- [`resampy=0.2.1`](http://resampy.readthedocs.io/en/latest/)
- [`six=1.12.0`](https://pythonhosted.org/six/)
- [`librosa=0.6.3=py_0`](https://librosa.github.io/librosa/)
- [`pysoundfile==0.9.0.post1`](https://pysoundfile.readthedocs.io/)
- [`ffmpeg=4.1.3`](https://ffmpeg.org/)

Loki Installation
-----------------

Once dependencies are installed, do:
```
cd build
source add_path.sh
```

The `add_path.sh` script will add the relevant directories to the PYTHONPATH variable.
It also sets the necessary environmental variables for finding the necessary checkpoint files.
It will also check for the required VGGish checkpoint file and `wget` it if it is not found.
This must be downloaded inside the build directory for the neural network classifier to work.
It can be found in TensorFlow checkpoint format at: [VGGish model checkpoint](https://storage.googleapis.com/audioset/vggish_model.ckpt).

The methods in the LOKI analysis package should then be usable as:

```
import loki

clips = loki.VideoClips(["example.mp4"])
```

Helper functions exist in `loki.functions.helper` which provide convenient functions for processing video files and outputting trained models.

Example
=======
To run the example and test the `loki` package, do:

```
cd example
python -m example_script
```
If there are no errors, it will:
1. Load the local .mp4 files.
2. Train a neural network classifier on the video data to identify interesting moments. In this case, interesting is when there is banging on the tin lid.
3. Perform inference on the training data and print out the confusion matrix.
4. Compute an Interest vs. Time on the test mp4 file.
5. Find the most interesting 1-second segment and 3-second segment in the test mp4 file.

Developer Notes
===============
The .gitignore file ignores all files by default. If you want to add a
new file or filetype to the repo, the .gitignore file must be amended.

Acknowledgements
================
This project was made as a consulting project at the Insight Artificial Intelligence Program.
I am grateful for the support and guidance the Insight community provided.
I also want to thank the company I consulted with, [Visor](https://visor.gg/), for providing video files to train and test the model.

This application uses Open Source components, specifically files contained in `loki/models/vggish_tensorflow/`. You can find the source code of their open source projects along with license information below. We acknowledge and are grateful to these developers for their contributions to open source.

Project: vggish https://github.com/tensorflow/models/tree/master/research/audioset/vggish

Copyright 2016 The TensorFlow Authors. All rights reserved.

License: Apache License 2.0 https://github.com/tensorflow/models/blob/master/LICENSE
