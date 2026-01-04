import pandas as pd
import os
import config

#関節角度の特徴量と加速度の特徴量を結合するスクリプト

# ファイルパスの設定 
#df_accel_sacr_path = config.df_accel_sacr
df_accel_sacr_path = config.df_accel_sacr_renamed
df_accel_rank_path = config.df_accel_rank
df_accel_rank2_path = config.df_accel_rank2
df_knee_vgrf_QC_path = config.df_knee_vgrf_QC
# 出力ファイルパスの設定 元のファイルと同じディレクトリ
output_folder = os.path.dirname(df_accel_sacr_path)
output_csv_path = os.path.join(output_folder, "merged_features_kinematics7.csv")

#データの読み込み 
try:
    df_accel_sacr = pd.read_csv(df_accel_sacr_path)
    df_accel_rank = pd.read_csv(df_accel_rank_path)
    df_accel_rank2 = pd.read_csv(df_accel_rank2_path)
    df_kinema = pd.read_csv(df_knee_vgrf_QC_path)
    print(" data loaded successfully.")
except FileNotFoundError as e:
    print(f" file not found: {e.filename}")
    # 続行不可のため終了
    exit()


# 関節角度，vGRFQC後のファイル用 キー列の作成/修正 
# Data 2の 'File' 列を、Data 1の 'file_name' と一致するように整形
# '_markers.trc' を削除し、'.c3d' に置き換える
df_kinema['file_name_key'] = (
    df_kinema['File']
    .str.replace('_markers.trc', '.c3d', regex=False)
)

# データの統合 
# SACR + RANK
df_accel = pd.merge(
    df_accel_sacr,
    df_accel_rank,
    on=['file_name', 'step_index'],
    how='left'
)

# + RANK2
df_accel = pd.merge(
    df_accel,
    df_accel_rank2,
    on=['file_name', 'step_index'],
    how='left'
)

df_merged = pd.merge(
    df_accel,
    df_kinema.drop(columns=['File']), # 元の 'File' 列は削除
    left_on='file_name',
    right_on='file_name_key',
    how='left' # 加速度データ(df_accel)をベースに結合
).drop(columns=['file_name_key']) # 結合に使ったキー列を削除


#  結果のCSVファイルへの書き出し 
df_merged.to_csv(output_csv_path, index=False)

print(f"\nsave:")
print(f"> {output_csv_path}")
print("-" * 30)


# 結果の確認表示 
print("## (df_merged.head())")
print(df_merged.head().to_markdown(index=False, numalign="left", stralign="left"))