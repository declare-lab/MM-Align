#!/bin/bash

path=$1
split=$2

for seed in 2020 2021 2022 2023 2024;do
group_id=`expr $seed - 2020`
for ratio in 0.1 0.5
do
    python split_dataset.py --data_path $path --seed $seed --group_id $group_id --complete_ratio $ratio --split $split
done
done