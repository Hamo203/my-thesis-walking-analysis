import numpy as np
import pandas as pd
import ezc3d
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis
from numpy import trapz
import os
import matplotlib.pyplot as plt
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
import config

# === フィルタ・特徴量関数 ===

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low', analog=False)
    return b, a

def lowpass_signal(x, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, x, axis=0)

def compute_acc_versions(pos, fs):
    # 位置を6Hzでフィルタリング
    pos_6Hz = lowpass_signal(pos, 6.0, fs)
    # 速度を計算 (pos_6Hzを1階微分)
    vel_6Hz = np.gradient(pos_6Hz, axis=0) * fs
    # 加速度を計算 (vel_6Hzを1階微分)
    acc_v3d = np.gradient(vel_6Hz, axis=0) * fs
    # 加速度を20Hzでフィルタリング
    acc_v3d = lowpass_signal(acc_v3d, 20.0, fs)
    return pos_6Hz, acc_v3d

def clean_ic_indices(ic_indices, min_interval=10):
    """RHSの重複（近すぎるイベント）を除去"""
    if len(ic_indices) < 2:
        return ic_indices
    cleaned = [ic_indices[0]]
    for i in range(1, len(ic_indices)):
        if ic_indices[i] - cleaned[-1] > min_interval: # 修正: ic_indices[i - 1] ではなく cleaned[-1] と比較
            cleaned.append(ic_indices[i])
    return cleaned

# tsfreshを組み込んだ特徴量抽出関数 

def extract_all_features(acc, ic_indices, fs, file_name):
    
    # 1. タイムセグメントをtsfresh用のDataFrameに変換
    tsfresh_data_list = []
    basic_feats = []
    
    for i in range(len(ic_indices)-1):
        s, e = ic_indices[i], ic_indices[i+1]
        seg = acc[s:e]
        
        if seg.size == 0:
            print(f"[Warning] Empty segment between {s} and {e} in file {file_name}")
            continue

        # tsfresh用データの準備 
        step_id = i
        mag = np.linalg.norm(seg, axis=1)
        time_index = np.arange(len(seg)) 
        
        df_step = pd.DataFrame({'id': step_id, 'time': time_index})
        
        df_step_x = df_step.assign(variable='acc_x', value=seg[:, 0])
        df_step_y = df_step.assign(variable='acc_y', value=seg[:, 1])
        df_step_z = df_step.assign(variable='acc_z', value=seg[:, 2])
        df_step_mag = df_step.assign(variable='acc_mag', value=mag)
        
        tsfresh_data_list.append(
            pd.concat([df_step_x, df_step_y, df_step_z, df_step_mag], ignore_index=True)
        )
        
        #  元の基本的な特徴量の抽出+PSDベースの特徴量
        feat = {
            'file_name': file_name,
            'step_index': i,
            'frames': e - s,
            'rms_mag': np.sqrt(np.mean(mag ** 2)),
            'total_power_mag': 0.0, 'peak_freq_mag': 0.0, 'median_freq_mag': 0.0,
            'ic_peak': 0
        }
        
        # PSDベースの特徴量
        try:
            freqs, Pxx = welch(mag, fs=fs, nperseg=len(mag), scaling='spectrum')
            total_power = trapz(Pxx, freqs)
            feat['total_power_mag'] = total_power
            peak_freq = freqs[np.argmax(Pxx)]
            feat['peak_freq_mag'] = peak_freq
            cumulative_power = np.cumsum(Pxx)
            median_freq_idx = np.searchsorted(cumulative_power, total_power / 2)
            median_freq = freqs[median_freq_idx] if median_freq_idx < len(freqs) else 0.0
            feat['median_freq_mag'] = median_freq
        except ValueError as e:
            feat.update({'total_power_mag': 0.0, 'peak_freq_mag': 0.0, 'median_freq_mag': 0.0})
            
        # IC直後0.15秒間のピーク
        win = seg[:int(0.15 * fs)]
        win_mag = np.linalg.norm(win, axis=1)
        feat['ic_peak'] = win_mag.max() if win_mag.size else 0
        
        basic_feats.append(feat)

    if not tsfresh_data_list:
        return pd.DataFrame()

    df_tsfresh_input = pd.concat(tsfresh_data_list, ignore_index=True)
    settings = EfficientFCParameters() 

    df_tsfresh_features = extract_features(
        df_tsfresh_input, 
        column_id='id', 
        column_sort='time', 
        column_value='value', 
        column_kind='variable',
        default_fc_parameters=settings,
        n_jobs=os.cpu_count() 
    )
    
    df_tsfresh_features = df_tsfresh_features.reset_index().rename(columns={'index': 'step_index'})
    df_basic = pd.DataFrame(basic_feats)
    
    df_all_features = pd.merge(df_basic, df_tsfresh_features, on='step_index', how='left')
    
    return df_all_features


if __name__ == '__main__':
    
    folder = config.folder_tsfresh
    output_csv = os.path.join(folder, "output_features_tsfresh2.csv")

    all_results = []
    
    # 最初のファイルのみを処理してプロットするために、ループを制限するフラグ
    plot_done = False 

    for file in os.listdir(folder):
        if not file.endswith(".c3d"):
            continue

        file_path = os.path.join(folder, file)
        print(f"\nProcessing: {file}")

        try:
            c3d = ezc3d.c3d(file_path)
        except Exception as e:
            print(f"[Error] Failed to read C3D file {file}: {e}")
            continue

        marker_name = "SACR"
        marker_labels = c3d['parameters']['POINT']['LABELS']['value']
        if marker_name not in marker_labels:
            print(f"{marker_name} not found in {file}")
            continue
        idx = marker_labels.index(marker_name)

        fs = float(c3d['parameters']['POINT']['RATE']['value'][0])
        pos_mm = c3d['data']['points'][:3, idx, :].T
        pos_m = pos_mm / 1000.0

        # 加速度を計算
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
        
        # RHSイベントをクリーニング
        ic_indices = clean_ic_indices(ic_indices, min_interval=int(0.05 * fs)) # 最小間隔を50msに変更
        print("IC indices (cleaned RHS):", ic_indices)

        #グラフにプロット
        if not plot_done:
            # タイムベクターの作成
            time_vector = np.arange(acc_v3d.shape[0]) / fs
            
            # グラフの作成
            plt.figure(figsize=(15, 8))
            plt.suptitle(f'SACR Acceleration and RHS Events: {file}', fontsize=14)
            
            axes_labels = ['X (0)', 'Y (1)', 'Z (2)']
            
            # X, Y, Z 軸の加速度をプロット
            for i in range(3):
                plt.subplot(3, 1, i + 1)
                plt.plot(time_vector, acc_v3d[:, i], label=f'Acc {axes_labels[i]}', linewidth=1)
                
                # RHSイベントを垂直線でプロット
                for idx_frame in ic_indices:
                    if idx_frame < len(time_vector):
                        plt.axvline(x=time_vector[idx_frame], color='r', linestyle='--', linewidth=1, label='RHS Event' if idx_frame == ic_indices[0] else "")
                        
                plt.title(f'Acceleration in {axes_labels[i]} Direction (m/s^2)')
                plt.xlabel('Time (s)')
                plt.ylabel('Acc (m/s^2)')
                plt.grid(True, linestyle=':', alpha=0.6)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # supertitleとの干渉を防ぐ
            plt.show()

            # 最初のファイルのみでプロットを停止
            # plot_done = True 
            # プロットを全ファイルで確認したい場合は上記のコメントアウトを外す
        
        
        # 特徴量抽出を実行 プロット後もデータ処理は続行
        df_feat = extract_all_features(acc_v3d, ic_indices, fs, file) 
        if not df_feat.empty:
            all_results.append(df_feat)
        
        # 最初のファイルのみ処理したい場合はここで break
        # break 


    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_csv(output_csv, index=False)
        print(f"\n Saved all features (including tsfresh) to {output_csv}")
    else:
        print(" No valid C3D files processed.")