#!/usr/bin/env bash

# specify which GPU to work on ...
export CUDA_VISIBLE_DEVICES=3

nvidia-smi
# obtain the directory the bash script is stored in


# DIR=$(cd $(dirname $0); pwd)
#--bind /data/corpora:/corpora
#--bind /data/users/maqsood/hf_cache:/cache
export HF_DATASETS_DOWNLOADED_DATASETS_PATH='/corpora/common_phone/'
export HF_DATASETS_CACHE='/cache'
python -u /nethome/mmshaik/Hiwi/common_phone_analysis/cp_feature_extractor.py