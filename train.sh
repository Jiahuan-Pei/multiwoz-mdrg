#!/usr/bin/env bash
mkdir -p logs
source activate multiwoz
TIMESTAMP=`date "+%Y%m%d%H%M%S"`
python2 -u train.py --data_dir='../multiwoz-moe/data' >> logs/laptop-$TIMESTAMP.train

