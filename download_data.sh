#!/bin/bash

save_dir="./DiffusionForensics"
mkdir -p "$save_dir"

urls=(
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00000-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00001-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00002-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00003-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00004-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00005-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00006-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00007-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00008-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00009-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00010-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00011-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00012-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00013-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00014-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00015-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00016-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00017-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00018-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00019-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00020-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00021-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00022-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00023-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/data-00024-of-00025.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/dataset_info.json"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/mapping.json"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/state.json"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DiffusionForensics/test.json"
)


for url in "${urls[@]}"
do
    file_name=$(basename "$url")
    wget -O "${save_dir}/${file_name}" "$url"
done



save_dir="./Ojha"
mkdir -p "$save_dir"

urls=(
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/Ojha/data-00000-of-00003.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/Ojha/data-00001-of-00003.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/Ojha/data-00002-of-00003.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/Ojha/dataset_info.json"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/Ojha/mapping.json"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/Ojha/state.json"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/Ojha/test.json"
)


for url in "${urls[@]}"
do
    file_name=$(basename "$url")
    wget -O "${save_dir}/${file_name}" "$url"
done


save_dir="./DIF"
mkdir -p "$save_dir"

urls=(
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00000-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00001-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00002-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00003-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00004-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00005-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00006-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00007-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00008-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00009-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00010-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00011-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00012-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00013-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00014-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00015-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00016-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00017-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00018-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00019-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00020-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00021-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00022-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00023-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00024-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/data-00025-of-00026.arrow"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/dataset_info.json"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/mapping.json"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/state.json"
    "https://huggingface.co/datasets/nebula/DF-arrow/resolve/main/DIF/test.json"
)


for url in "${urls[@]}"
do
    file_name=$(basename "$url")
    wget -O "${save_dir}/${file_name}" "$url"
done













