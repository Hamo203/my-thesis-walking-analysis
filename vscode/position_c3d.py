import ezc3d
import numpy as np
import pandas as pd

import config

# 基本的なC3Dファイルの読み込みとデータ抽出のコード
file_path = config.c3d_cal_c3d
c3d = ezc3d.c3d(file_path)

# 例えば、右膝のマーカー "RKNE" の位置データを取得する例
marker_name = "RKNE"
    
marker_labels = c3d['parameters']['POINT']['LABELS']['value']
if marker_name not in marker_labels:
    print(f"{marker_name} not found in {file_path}")
idx = marker_labels.index(marker_name)

fs = float(c3d['parameters']['POINT']['RATE']['value'][0])
# Visual 3D の x,y,z軸の順序で位置データを取得
pos_mm = c3d['data']['points'][:3, idx, :].T

# 例えば z が 7+0.2e -> 700mm になるため、単位を m に変換
pos_m = pos_mm / 1000.0

print("Position data (m):", pos_m)