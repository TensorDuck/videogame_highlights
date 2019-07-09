#set the necessary path variables
cd ..
CURRENT=`pwd`
cd build
export PYTHONPATH="$CURRENT:$PYTHONPATH"
export SOUNDEMBEDDINGS="$CURRENT/build/vggish_model.ckpt"

#download the vggish model checkpoint file if it does not already exist
if test -f "vggish_model.ckpt"; then
echo vggish_model.ckpt is already downloaded.
else
echo downloading vggish_model.ckpt
wget https://storage.googleapis.com/audioset/vggish_model.ckpt
fi
