#!/bin/bash
export BASE_PATH=/DATA2/disk1/wangyongbo/ms_marco/data

python passage_ranking.py --input_file $BASE_PATH/lines_dev_v2.1.json --output_file $BASE_PATH/lines_dev_v2.1_ranked.json