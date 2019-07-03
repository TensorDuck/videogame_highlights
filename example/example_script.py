import os
import subprocess as sb
import time
import numpy
import matplotlib.pyplot as plt

import loki

if __name__ == "__main__":
    cwd = os.getcwd()

    print("##################################################")
    print("Load Files")
    print("##################################################")
    #Best practice is to give full paths
    train_files = [f"{cwd}/train_good.mp4", f"{cwd}/train_bad.mp4"]
    test_files = [f"{cwd}/test.mp4"]

    train_targets = [1, 0]

    train_videos = loki.VideoClips(train_files)
    test_videos = loki.VideoClips(test_files)

    #get a trained neural network classifier
    print("##################################################")
    print("Begin Training")
    print("##################################################")
    nnclass = loki.helper.train_classifier(train_videos, train_targets, test_clips=train_videos, test_targets=train_targets, n_epochs=100, class_weights=None, batch_size=None)
    #save the neural network
    nnclass.save("example_nn")

    #perform inference on the training data
    train_audio = train_videos.compute_audio_waveform(mono=True)
    inferred = nnclass.infer(train_audio)
    loki.evaluation.print_confusion_matrix(train_targets, inferred)

    print("##################################################")
    print("Analyze test.mp4")
    print("##################################################")

    #single channel for Loki
    test_audio = test_videos.compute_audio_waveform(mono=True)
    #interest at each time-step
    x_trace, y_trace = nnclass.get_trace(test_audio)
    n_trace = len(x_trace[0])
    print("Time      Interest")
    print("------------------")
    for i in range(n_trace):
        print(f"{x_trace[0][i]:.2f}      {y_trace[0][i]:.4f}")

    print("##################################################")
    print("Find The Most Interesting 1-Second Clip from test.mp4")
    print("##################################################")
    #Use helper function to find the most relevant 1-second section
    results = loki.helper.find_best_clip(test_files, 1, nn_checkpoint="example_nn")
    print(results)
