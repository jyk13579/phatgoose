export MM_ROOT=`pwd`
export PYTHONPATH=$MM_ROOT:$PYTHONPATH
export PYTHON_EXEC=python
mkdir -p ~/.cache/phatgoose/
export HUGGINGFACE_HUB_CACHE=/mnt/nas/jiyeon/.cache/huggingface/hub #~/.cache/phatgoose/
export TRANSFORMERS_CACHE=/mnt/nas/jiyeon/.cache/huggingface #~/.cache/phatgoose/
export HF_HOME=/mnt/nas/jiyeon/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=phatgoose
