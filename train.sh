#!/usr/bin/env bash
mkdir -p logs
source activate multiwoz
TIMESTAMP=`date "+%Y%m%d%H%M%S"`
python2 -u train.py >> logs/laptop-$TIMESTAMP.train

