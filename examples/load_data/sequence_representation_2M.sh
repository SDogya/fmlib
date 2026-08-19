#!/bin/bash

set -x

local_path="/home/datalab/nfs/data/sequence_representation_2M"
remote_path="hdfs://arnsdpsbx/user/team/team_ai_avatar/avatar_fm/examples/next_event_prediction/processed_sequences"

if [ -d ${local_path} ]; then
    echo "Removing \"sequence_representation_2M\" data..."
    rm -rf ${local_path}
fi

echo "Loading \"sequence_representation_2M\" data..."
mkdir -p ${local_path}
hadoop fs -get ${remote_path} ${local_path}

set +x
