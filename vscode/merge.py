import pandas as pd
import os
import config

# ファイルパスの設定 
data1_path = config.data1_path
data2_path = config.data2_path
# 出力ファイルパスの設定 元のファイルと同じディレクトリ
output_folder = os.path.dirname(data1_path)
output_csv_path = os.path.join(output_folder, "merged_features_kinematics4.csv")

#データの読み込み 
try:
    df_accel = pd.read_csv(data1_path)
    df_kinema = pd.read_csv(data2_path)
    print(" data loaded successfully.")
except FileNotFoundError as e:
    print(f" file not found: {e.filename}")
    # 続行不可のため終了
    exit()


#キー列の作成/修正 
# Data 2の 'File' 列を、Data 1の 'file_name' と一致するように整形
# '_markers.trc' を削除し、'.c3d' に置き換える
df_kinema['file_name_key'] = (
    df_kinema['File']
    .str.replace('_markers.trc', '.c3d', regex=False)
)

# データの統合 
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