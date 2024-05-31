#!/bin/bash
# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))
exp_config="${exp_dir}"/exp_config.json

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

# cd $work_dir/modules/monotonic_align
# mkdir -p monotonic_align
# python setup.py build_ext --inplace
cd $work_dir

mfa_dir=$work_dir/pretrained/mfa
echo $mfa_dir

######## Features Extraction ###########
# if [ ! -d "$mfa_dir/montreal-forced-aligner" ]; then
#     bash ${exp_dir}/prepare_mfa.sh
# fi
CUDA_VISIBLE_DEVICES=$gpu python "${work_dir}"/bins/tts/preprocess.py \
    --config=$exp_config \
    --num_workers=4 || exit 1
    # --prepare_alignment=False
