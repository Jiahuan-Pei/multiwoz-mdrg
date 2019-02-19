#!/usr/bin/env bash
source activate multiwoz
TIMESTAMP=`date "+%Y%m%d%H%M%S"`
python2 -u test.py >> logs/laptop.test