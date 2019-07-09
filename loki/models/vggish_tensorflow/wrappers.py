"""This file contains wrappers for the VGGish methods

Large parts of this file was copied from the colab for the VGGish
method, see:
https://colab.research.google.com/drive/1TbX92UL9sYWbdwdGE0rJ9owmezB-Rl1C
"""
import tensorflow as tf

from . import vggish_slim
from . import vggish_params
from . import vggish_input

def CreateVGGishNetwork(sess, checkpoint_path, hop_size=0.96):   # Hop size is in seconds.
    """Define VGGish model, load the checkpoint, and return a dictionary
     that points to the different tensors defined by the model.
    """
    vggish_slim.define_vggish_slim()
    vggish_params.EXAMPLE_HOP_SECONDS = hop_size

    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

    features_tensor = sess.graph.get_tensor_by_name(
      vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
      vggish_params.OUTPUT_TENSOR_NAME)

    layers = {'conv1': 'vggish/conv1/Relu',
            'pool1': 'vggish/pool1/MaxPool',
            'conv2': 'vggish/conv2/Relu',
            'pool2': 'vggish/pool2/MaxPool',
            'conv3': 'vggish/conv3/conv3_2/Relu',
            'pool3': 'vggish/pool3/MaxPool',
            'conv4': 'vggish/conv4/conv4_2/Relu',
            'pool4': 'vggish/pool4/MaxPool',
            'fc1': 'vggish/fc1/fc1_2/Relu',
            'fc2': 'vggish/fc2/Relu',
            'embedding': 'vggish/embedding',
            'features': 'vggish/input_features',
         }
    g = tf.get_default_graph()
    for k in layers:
        layers[k] = g.get_tensor_by_name( layers[k] + ':0')

    return {'features': features_tensor,
          'embedding': embedding_tensor,
          'layers': layers,
         }

def EmbeddingsFromVGGish(sess, vgg, x, sr):
    '''Run the VGGish model, starting with a sound (x) at sample rate
    (sr). Return a dictionary of embeddings from the different layers
    of the model.'''
    # Produce a batch of log mel spectrogram examples.
    input_batch = vggish_input.waveform_to_examples(x, sr)
    # print('Log Mel Spectrogram example: ', input_batch[0])

    layer_names = vgg['layers'].keys()
    tensors = [vgg['layers'][k] for k in layer_names]

    results = sess.run(tensors,
                     feed_dict={vgg['features']: input_batch})

    resdict = {}
    for i, k in enumerate(layer_names):
        resdict[k] = results[i]

    return resdict
