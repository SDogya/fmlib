#!/bin/bash

set -x

local_path="/home/datalab/nfs/data/feature_transformer"
remote_path="hdfs://arnsdpsbx/user/team/team_ai_avatar/avatar_fm/examples/campaign_demo/td_processed"

for split in "train" "valid" "test"; do
    split_path="${local_path}/${split}"
    if [ -d ${split_path} ]; then
        echo "Removing \"${split}\" data..."
        rm -rf ${split_path}
    fi
    echo "Loading \"${split}\" data..."
    mkdir -p ${split_path}
    hadoop fs -get ${remote_path}/${split} ${local_path}
done

set +x
