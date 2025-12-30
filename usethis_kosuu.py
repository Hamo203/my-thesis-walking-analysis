# use_thisになっている行数の確認
import pandas as pd
import os
import config

df_merged = pd.read_csv(config.output_csv_usethis_kosuu)

# use_this カラムが 1 の行をフィルタリングし、その行数をカウントします。
count_use_this_1 = df_merged[df_merged['use_this'] == 1].shape[0]

print(f"`use_this = 1` : **{count_use_this_1}** rows found.")