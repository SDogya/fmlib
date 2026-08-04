with open("/home/datalab/nfs/fs_benchmark/features_config.yaml", "r") as f:
    df_features = yaml.safe_load(f)
stop_list = set(["target_attr_1", "target_attr_2", "target_attr_3", "bucket_num", "month_part", "split_type", "report_month"])
features = df_features['cat_cols'] + df_features['num_cols']

with open("TD/stage_0_DROPPED.txt", "r") as f:
    drop_features = f.read().strip().split("\n")
drop_features = [i for i in drop_features if i not in stop_list]
with open("/home/datalab/projects/avatar/avatards/zaripov_glebasta/fs_drop_features/TD_stage_0.txt", "w") as f:
    f.writelines(line + '\n' for line in drop_features)
len(drop_features)

with open("PD/stage_0_1_SELECTED.txt", "r") as f:
    selected_features = f.read().strip().split("\n")
selected_features = [i for i in selected_features if i not in stop_list]
drop_features = list(set(features) - set(selected_features))
with open("/home/datalab/projects/avatar/avatards/zaripov_glebasta/fs_drop_features/PD_stage_0_1.txt", "w") as f:
    f.writelines(line + '\n' for line in drop_features)
len(drop_features)



# import subprocess as sss

# data_sets = [PD,TD,SA_ERKC]

# dfs_path = /user/team/team_ai_avatar/ds/bogachev/featureselection/pipeline/stage_0_a_stage_0_1/

# feature_pathe = 

