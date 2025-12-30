# 必要ライブラリ
import numpy as np
import pandas as pd
import ezc3d
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import config
def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    #フィルタ次数(order), 正規化カットオフ周波数(cutoff/nyq) ,btypeフィルタのタイプ, analog=False(デジタルフィルタ)
    b, a = butter(order, cutoff/nyq, btype='low', analog=False)
    return b, a

# ローパスフィルタ適用をするための関数
def lowpass_signal(x, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    #位相遅延を避けるために双方向フィルタリング
    # filtfilt for zero-phase
    return filtfilt(b, a, x, axis=0)

def compute_acc_versions(pos, fs):
    """6Hz位置フィルタ→20Hz加速度フィルタ（Visual3D再現）と生微分版を両方計算"""
    #pos_6hz: 6Hzローパスフィルタ後の位置データ
    pos_6Hz = lowpass_signal(pos, 6.0, fs)
    # 速度・加速度計算
    vel_6Hz = np.gradient(pos_6Hz, axis=0) * fs
    acc_v3d = np.gradient(vel_6Hz, axis=0) * fs
    #acc_v3d = lowpass_signal(acc_v3d, 20.0, fs)  
    return pos_6Hz, acc_v3d

def extract_basic_features(acc, ic_indices, fs):
    # acc: (n_frames,3) ; ic_indices: list of frame indices marking initial contacts
    feats = []
    for i in range(len(ic_indices)-1):
        # LHSから次のLHSのインデックス
        s,e = ic_indices[i], ic_indices[i+1]
        # plt.plot(np.linalg.norm(acc_v3d[s:e], axis=1))
        # plt.title("Left Step (LHS to next LHS)")
        # plt.xlabel("Frame")
        # plt.ylabel("Acceleration Magnitude (m/s^2)")
        # plt.show()
        seg = acc[s:e]
        mag = np.linalg.norm(seg, axis=1)
        feat = {
            'step_index': i,  # ステップインデックス->0始まり
            'frames': e - s, # フレーム数
            'mean_x': seg[:, 0].mean(), 'std_x': seg[:, 0].std(),
            'mean_y': seg[:, 1].mean(), 'std_y': seg[:, 1].std(),
            'mean_z': seg[:, 2].mean(), 'std_z': seg[:, 2].std(),
            'mean_mag': mag.mean(), 'std_mag': mag.std(),
            'max_mag': mag.max(), 'min_mag': mag.min(),
            'rms_mag': np.sqrt(np.mean(mag ** 2)),
            'skew_mag': skew(mag), 'kurt_mag': kurtosis(mag)
        }
        # IC直後150msのピーク値
        win = seg[:int(0.15 * fs)]
        win_mag = np.linalg.norm(win, axis=1)
        feat['ic_peak'] = win_mag.max() if win_mag.size else 0
        feats.append(feat)
    return pd.DataFrame(feats)

# c3d 読み込み
file_path = config.c3d_calacc
c3d = ezc3d.c3d(file_path)

# マーカー名に合わせて変更
marker_name = "SACR"
# find marker index
marker_labels = c3d['parameters']['POINT']['LABELS']['value']
idx = marker_labels.index(marker_name)


# モーションキャプチャーのサンプリング周波数 (Hz) ->200Hz
fs = float(c3d['parameters']['POINT']['RATE']['value'][0])



markers = c3d['data']['points']  # shape: (4, n_markers72, n_frames208) 
pos_mm = markers[:3, idx, :].T  # (n_frames, 3)
pos_m = pos_mm / 1000.0 #visual3dはC3Dにmmで保存するのが標準仕様なのであわせる
print("pos_m shape:", pos_m.shape)
print("pos_m (first 5 rows):")
print(pos_m[:5])
print("pos_m min / max / mean per axis:")
print(np.min(pos_m, axis=0), np.max(pos_m, axis=0), np.mean(pos_m, axis=0))


#フィルタ処理確認
#pos_f: 6Hzローパスフィルタ後の位置データ
pos_f, acc_v3d= compute_acc_versions(pos_m, fs)
# pos_f (6Hz処理後) と pos_m 差分（確認用）
diff = pos_m - pos_f
print("pos_m - pos_f (first 10):")
print(diff[:10])
print("max abs diff:", np.max(np.abs(diff)))

# 加速度確認
print("acc_v3d shape:", acc_v3d.shape)
print("acc_v3d (first 5 rows):")
print(acc_v3d[:5])



# イベント情報の確認 
event_labels = c3d['parameters']['EVENT']['LABELS']['value']
event_times = c3d['parameters']['EVENT']['TIMES']['value'][1]  # 秒
print("EVENT_LABELS:", event_labels)
print("EVENT_TIMES:", event_times)

# RHSイベント抽出（右接地）
ic_times = [t for label, t in zip(event_labels, event_times) if label == 'RHS']
ic_indices = [int(t * fs) for t in ic_times]
print("IC indices (RHS):", ic_indices)

# 特徴量を抽出 
features_v3d = extract_basic_features(acc_v3d, ic_indices, fs)

save_path = config.save_calacc
features_v3d.to_csv(save_path, index=False)
print(f"\nSaved features_v3d  to {save_path}")