#!/usr/bin/env bash
mkdir -p logs
source activate multiwoz
TIMESTAMP=`date "+%Y%m%d%H%M%S"`
python2 -u train.py --data_dir='../multiwoz1-moe/data' --debug=True --max_epochs=2 >> logs/laptop-$TIMESTAMP.train

