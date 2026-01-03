import numpy as np
import pandas as pd
import ezc3d
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis
from numpy import trapz
import os
import config
# handcrafted特徴量のみを実行するバージョン
#フィルタ・特徴量関数
def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low', analog=False)
    return b, a

def lowpass_signal(x, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, x, axis=0)

def compute_acc_versions(pos, fs):
    pos_6Hz = lowpass_signal(pos, 6.0, fs)
    vel_6Hz = np.gradient(pos_6Hz, axis=0) * fs
    acc_v3d = np.gradient(vel_6Hz, axis=0) * fs
    #acc_v3d = lowpass_signal(acc_v3d, 20.0, fs)
    return pos_6Hz, acc_v3d

def clean_ic_indices(ic_indices, min_interval=10):
    """RHSの重複（近すぎるイベント）を除去"""
    if len(ic_indices) < 2:
        return ic_indices
    cleaned = [ic_indices[0]]
    for i in range(1, len(ic_indices)):
        if ic_indices[i] - ic_indices[i - 1] > min_interval:
            cleaned.append(ic_indices[i])
    return cleaned

def extract_basic_features(acc, ic_indices, fs, file_name, marker_name):
    feats = []
    for i in range(len(ic_indices)-1):
        s, e = ic_indices[i], ic_indices[i+1]
        seg = acc[s:e]
        if seg.size == 0:
            print(f"[Warning] Empty segment between {s} and {e} in file {file_name}")
            continue
            
        mag = np.linalg.norm(seg, axis=1)
        suffix = f"_{marker_name.lower()}"
        feat = {
            'file_name': file_name,
            'step_index': i,#仮に歩数が複数になったとき用
            'frames': e - s,
            f'mean_x{suffix}': seg[:, 0].mean(), f'std_x{suffix}': seg[:, 0].std(),
            f'mean_y{suffix}': seg[:, 1].mean(), f'std_y{suffix}': seg[:, 1].std(),
            f'mean_z{suffix}': seg[:, 2].mean(), f'std_z{suffix}': seg[:, 2].std(),
            f'mean_mag{suffix}': mag.mean(), f'std_mag{suffix}': mag.std(),
            f'max_mag{suffix}': mag.max(), f'min_mag{suffix}': mag.min(),
            f'rms_mag{suffix}': np.sqrt(np.mean(mag ** 2)),
            f'skew_mag{suffix}': skew(mag), f'kurt_mag{suffix}': kurtosis(mag)
        }
        try:
            # Welch法を用いて計算したパワースペクトル密度(PSD) の値に基づき算出された特徴量の計算
            # fs: サンプリング周波数, nperseg: セグメント長
            freqs, Pxx = welch(mag, fs=fs, nperseg=len(mag), scaling='spectrum')
            # 1. 総パワー (エネルギー総量)
            total_power = trapz(Pxx, freqs)
            feat[f'total_power_mag{suffix}'] = total_power
            # 2. ピーク周波数 (最大パワー周波数)
            peak_freq = freqs[np.argmax(Pxx)]
            feat[f'peak_freq_mag{suffix}'] = peak_freq
            # 3. メジアン周波数 (パワー半分の周波数)
            cumulative_power = np.cumsum(Pxx)
            median_freq_idx = np.searchsorted(cumulative_power, total_power / 2)
            median_freq = freqs[median_freq_idx] if median_freq_idx < len(freqs) else 0.0
            feat[f'median_freq_mag{suffix}'] = median_freq
            
        except ValueError as e:
            # FFT計算が不可能な場合 (セグメントが短すぎるなど)
            print(f"[Warning] FFT failed for step {i} in {file_name}: {e}")
            feat[f'total_power_mag{suffix}'] = 0.0
            feat[f'peak_freq_mag{suffix}'] = 0.0
            feat[f'median_freq_mag{suffix}'] = 0.0

        # --- IC直後0.15秒間のピーク ---
        win = seg[:int(0.15 * fs)]
        win_mag = np.linalg.norm(win, axis=1)
        feat[f'ic_peak{suffix}'] = win_mag.max() if win_mag.size else 0
        feats.append(feat)
    return pd.DataFrame(feats)



folder = config.folder_calc_all
output_csv = os.path.join(folder, "output_features_all_RAN2.csv")

all_results = []

for file in os.listdir(folder):
    if not file.endswith(".c3d"):
        continue

    file_path = os.path.join(folder, file)
    print(f"Processing: {file}")

    c3d = ezc3d.c3d(file_path)

    marker_name = "RAN2"
    marker_labels = c3d['parameters']['POINT']['LABELS']['value']
    if marker_name not in marker_labels:
        print(f"{marker_name} not found in {file}")
        continue
    idx = marker_labels.index(marker_name)

    fs = float(c3d['parameters']['POINT']['RATE']['value'][0])
    pos_mm = c3d['data']['points'][:3, idx, :].T
    pos_m = pos_mm / 1000.0

    pos_f, acc_v3d = compute_acc_versions(pos_m, fs)

    # イベント情報
    if 'EVENT' not in c3d['parameters']:
        print(f" No EVENT found in {file}")
        continue

    event_labels = c3d['parameters']['EVENT']['LABELS']['value']
    event_times = c3d['parameters']['EVENT']['TIMES']['value'][1]
    ic_times = [t for label, t in zip(event_labels, event_times) if label == 'RHS']
    ic_indices = [int(t * fs) for t in ic_times]
    if len(ic_indices) < 2:
        print(f"Not enough RHS events in {file}")
        continue
    
    ic_indices = clean_ic_indices(ic_indices, min_interval=10)
    print("IC indices (cleaned RHS):", ic_indices)


    df_feat = extract_basic_features(acc_v3d, ic_indices, fs, file, marker_name)
    all_results.append(df_feat)


if all_results:
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(output_csv, index=False)
    print(f"\n Saved all features to {output_csv}")
else:
    print(" No valid C3D files processed.")