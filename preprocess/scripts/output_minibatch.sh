#!/bin/bash

for i in 00
do
    python /workspace/pmod/preprocess/utils/output_minibatch.py \
        -hdf5 /workspace/pmod/datasets/kitti/train_500/kitti360_seq${i}_train.hdf5 \
        -c /workspace/pmod/config/kitti360-5class.json
done

