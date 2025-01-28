#!/bin/bash

python /workspace/pmod/preprocess/utils/set_iopaintimg.py \
    -hdf5 /workspace/pmod/datasets/kitti/train_500/kitti360_seq00_train.hdf5 \
    -c /workspace/pmod/config/kitti360-5class.json \
    -d /workspace/pmod/preprocess/utils/output/iopaint/kitti/lama/seq00



