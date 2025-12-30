# 20Hzローパスフィルタ前後の加速度の周波数スペクトルを比較するコード
# 結論:20Hzローパスフィルタをかけてもあまり変化がないので使う必要がないかも
import ezc3d
import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
import config
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
    acc_v3d = lowpass_signal(acc_v3d, 20.0, fs)
    return pos_6Hz, acc_v3d


def plot_fft(sig, fs):
    sig = sig - np.mean(sig, axis=0) # 直流成分の除去
    N = sig.shape[0]
    yf = np.abs(rfft(sig[:,0]))
    xf = rfftfreq(N, 1/fs)
    return xf, yf

if __name__ == '__main__':
    c3d = ezc3d.c3d(config.c3d_to_compare_filter)
    marker_name = "SACR"
    marker_labels = c3d['parameters']['POINT']['LABELS']['value']
    if marker_name not in marker_labels:
        print(f"{marker_name} not found in c3d file")
    idx = marker_labels.index(marker_name)

    fs = float(c3d['parameters']['POINT']['RATE']['value'][0])
    pos_mm = c3d['data']['points'][:3, idx, :].T
    pos_m = pos_mm / 1000.0
    
    pos_6Hz, acc_20Hz = compute_acc_versions(pos_m, fs)

    vel = np.gradient(pos_6Hz, axis=0) * fs
    acc_raw = np.gradient(vel, axis=0) * fs

    xf1, yf1 = plot_fft(acc_raw, fs)
    xf2, yf2 = plot_fft(acc_20Hz, fs)

    plt.plot(xf1, yf1, label='Before 20Hz LPF')
    plt.plot(xf2, yf2, label='After 20Hz LPF')
    plt.xlim(0, 50)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("FFT Magnitude")
    plt.title("Comparison of Frequency Spectra Before and After 20 Hz Low-Pass Filtering")
    plt.legend()
    plt.grid()
    plt.show()