import pandas as pd
import config
import os

df_accel_sacr_path = config.df_accel_sacr
# suffix を付けたくない列
exclude_cols = ['file_name', 'step_index', 'frames']
df = pd.read_csv(df_accel_sacr_path)
df = df.rename(
    columns=lambda c: c if c in exclude_cols else f"{c}_sacr"
)
# 出力ファイルパスの設定 元のファイルと同じディレクトリ
output_folder = os.path.dirname(df_accel_sacr_path)
df.to_csv(os.path.join(output_folder, "output_features_all_sacr_renamed.csv"), index=False)