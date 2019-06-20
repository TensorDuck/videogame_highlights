cd ..
CURRENT=`pwd`
cd build
export PYTHONPATH="$CURRENT:$PYTHONPATH"
#export SOUNDEMBEDDINGS="$CURRENT/loki/models/vggish_tensorflow/vggish_model.ckpt"
export SOUNDEMBEDDINGS="$CURRENT/build/vggish_model.ckpt"
