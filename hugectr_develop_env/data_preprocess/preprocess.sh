#!/bin/bash

if [[ $# -ne 3 ]]; then
  echo "usage: preprocess.sh [NAME] [NORMALIZE_DENSE] [FEATURE_CROSS]"
  exit 2
fi

tar zxvf dac.tar.gz
mkdir -p $1_data
shuf train.txt > train.shuf.txt
python3 preprocess_multithreading.py --src_csv_path=train.shuf.txt --dst_csv_path=$1_data/train.out.txt --normalize_dense=$2 --feature_cross=$3
head -n 36672493 $1_data/train.out.txt > $1_data/train
tail -n 9168124 $1_data/train.out.txt > $1_data/valtest
head -n 4584062 $1_data/valtest > $1_data/val
tail -n 4584062 $1_data/valtest > $1_data/test
