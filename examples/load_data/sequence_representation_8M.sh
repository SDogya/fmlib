#!/bin/bash

set -x

local_path="/home/datalab/nfs/data/sequence_representation_8M"
remote_path="hdfs://arnsdpsbx/user/team/team_ai_avatar/ds/rusakov/ssl_training"

for split in "middle/small/mini_small" "valid/mini_valid"; do
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
