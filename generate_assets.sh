#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0
# export TORCH_USE_CUDA_DSA

# 读取文件内容
while IFS= read -r line
do
    # 去除行尾的换行符，并分割行以获取最后一个单词
    factory_name=$(echo "$line" | awk -F'[.]' '{print $NF}' | sed 's/Factory//')
    # 构造命令行命令
    command="python -m infinigen_examples.generate_individual_assets --output_folder data_v1 -f ${factory_name}Factory -n 8 --save_blend"
    # 执行命令
    echo "Executing: $command"
    eval $command
done < "tests/assets/list_indoor_meshes_2.txt"

echo "All assets have been generated."