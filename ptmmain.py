#%%
import wave
import sys
import numpy as np
import matplotlib.pyplot as plt
import array
import glob
from playsound import playsound #raspiではsimpleaudio
import librosa
import librosa.feature
from scipy import interpolate
from scipy import signal
import cv2
from collections import deque


#IOモジュール


def read_wave(filename):
    wf = wave.open(filename, "r")
    wv = wf.readframes(wf.getnframes())
    wv = np.frombuffer(wv, dtype="int16")
    if wf.getnchannels() == 2:
        wv = wv[::2]
    return wf, wv

def printWaveInfo(wf):
    """WAVEファイルの情報を取得"""
    print("チャンネル数 : "+ str(wf.getnchannels()))
    print("サンプル幅 : "+ str(wf.getsampwidth()))
    print("サンプルレート : "+ str(wf.getframerate()))
    print("フレーム数 : "+ str(wf.getnframes()))
    print("総パラメータ（一括表示用） : "+ str(wf.getparams()))
    print("再生時間 : "+ str(float(wf.getnframes()) / wf.getframerate()))

def plot_wave(x):
    plt.plot(x, label="wave")
    plt.legend()
    plt.show()
    tmp = 1 + 1

#ndarray(shape:(-1,), monoral frames) -> filename(.wav)
def write_wave(filename, data, fs=48000):
    writewave = wave.Wave_write(filename)
    writewave.setparams((
        1,                      #channel
        2,                      #byte width
        fs,                     #sampling rate
        len(data),              #number of frames
        'NONE', 'not compressed'#no compression
    ))
    writewave.writeframes(array.array('h', data).tostring())
    writewave.close()


#あ.png読み込み


a01 = np.array(cv2.imread("./data/a_imgs/index.png", 0))
#a01 = np.array(cv2.imread("./data/a_imgs/hpan.jpg", 0))
#a01 = np.array(cv2.imread("./data/a_imgs/a01.png", 0))
#a01 = np.array(Image.open("./data/a_imgs/a01.png"))
#print(a01.shape)
#print(a01)
#plt.imshow(a01, cmap="gray")
#print(np.unique(a01))
#01 = a01/255.0


#edge検出


b01 = np.zeros_like(a01)
#plt.imshow(b01, cmap = "gray")
#plt.show()
#graymap = np.array([[0,   50,  100],
#                    [255, 200, 150]])
#plt.imshow(graymap, cmap="gray")
#黒と白の境目：100あたり？
#for i in range(a01.shape[0]):
#    for j in range(a01.shape[1]):
#        if a01[i][j] < 100:
#            b01[i][j] = 255
#plt.imshow(b01, cmap="gray")

def is_white(a):
    return a > 150

dir = np.array([-1, 0, 1])
for i in range(a01.shape[0]):
    for j in range(a01.shape[1]):
        b01[i][j] = 255
        if is_white(a01[i][j]):
            flag = False
            for k in range(3):
                for l in range(3):
                    if k == 1 & l == 1:
                        continue
                    nx = i + dir[k]
                    ny = j + dir[l]
                    if nx < 0 or nx >= a01.shape[0] or ny < 0 or ny >= a01.shape[1]:
                        continue
                    flag = flag or not is_white(a01[nx][ny])
            if flag:
                b01[i][j] = 0

plt.imshow(b01, cmap="gray")
plt.show()
#print("a ここまでがとても遅い")

def is_edge(a):
    return a == 0

is_used = np.zeros_like(b01)
edgewave = np.zeros((100000,3), dtype=int)
numb = 0
for i in range(b01.shape[0]):
    for j in range(b01.shape[1]):
        if not is_edge(b01[i][j]) or is_used[i][j]:
            continue
        deq = deque()
        deq.append((i,j,0))
        is_used[i][j] = True
        while len(deq) > 0:
            cur = deq.pop()
            cx = cur[0]
            cy = cur[1]
            cdist = cur[2]
            edgewave[numb] = [cur[0],cur[1],cur[2]]
            numb = numb + 1
            for k in range(3):
                for l in range(3):
                    if k == l:
                        continue
                    nx = cx + dir[k]
                    ny = cy + dir[l]
                    if (nx < 0) or (nx >= b01.shape[0]) or (ny < 0) or (ny >= b01.shape[1]):
                        continue
                    if is_edge(b01[nx][ny]) and not is_used[nx][ny]:
                        deq.append((nx, ny, cdist + 1))
                        is_used[nx][ny] = True

#print(edgewave.shape)
#for i in range(20):#3470
#    print(edgewave[i + 2500][0:3])
edgewave = edgewave[:numb]
#xx = np.squeeze(edgewave[:numb,1:2])
#yy = np.squeeze(edgewave[:numb,0:1])
#xx = xx[np.where(xx != 0)]
#yy = yy[np.where(yy != 0)]
#print(xx)
#plt.plot(xx, yy, '.')
#plt.plot(np.arange(len(xx)), xx, '.')
#plt.plot(np.arange(len(yy)), yy, '.')


#境界のx,yを抽出したので、なめらかにする。隣接マスでないときの処理は前のマスに -1, +0, +1


def difto01(b):
    if b > 0:
        return 1
    elif b < 0:
        return -1
    else :
        return 0

bias = np.array([0,0])
for i in range(len(edgewave)):
    if i == 0:
        continue
    if abs(edgewave[i-1][0] - edgewave[i][0]) + abs(edgewave[i-1][1] - edgewave[i][1]) > 2:
        bias = edgewave[i-1,0:2] - edgewave[i,0:2]
        edgewave[i,0:2] = edgewave[i-1,0:2] + np.array([difto01(edgewave[i][0] - edgewave[i-1][0]), difto01(edgewave[i][1] - edgewave[i-1][1])])
    else :
        edgewave[i,0:2] = edgewave[i,0:2] + bias
    


#なめらかになったので中心化


xave = np.average(edgewave[:,0:1]).astype(int)
yave = np.average(edgewave[:,1:2]).astype(int)
#print(xave)
#print(yave)
edgewave[:,0:2] = edgewave[:, 0:2] - np.array([xave, yave])
#plt.plot(np.arange(len(edgewave)), np.squeeze(edgewave[:,0:1]), '.')
#plt.show()
#plt.plot(np.arange(len(edgewave)), np.squeeze(edgewave[:,1:2]), '.')
#plt.show()


#音に変換


wave1 = np.squeeze(edgewave[:,0:1])
wave2 = np.squeeze(edgewave[:,1:2])
#plot_wave(wave1)
#plot_wave(wave2)


#直線を用いて端を揃える


def connectLR(w):
    x1 = len(w) - 1
    y1 = w[0] - w[x1]
    a = y1/x1
    line = a * np.arange(len(w))
    w = w + line
    return w

wave1 = connectLR(wave1)
wave2 = connectLR(wave2)
#plot_wave(wave1)


#中心化


wave1 = wave1 - wave1[0]
wave2 = wave2 - wave2[0]
#plot_wave(wave1)


#平滑化


def heikatuka(w1):
    mergin = 50
    for i in range(mergin, len(w1) - mergin):
        w1[i] = np.mean(w1[i - mergin: i + mergin])
    return w1

wave1 = heikatuka(wave1)
wave2 = heikatuka(wave2)
#plot_wave(wave1)
#plot_wave(wave2)


#補完してダウンサンプリング


f = interpolate.interp1d(np.arange(len(wave1)), wave1, kind='cubic')
f2 = interpolate.interp1d(np.arange(len(wave2)), wave2, kind='cubic')
wave1 = f(np.arange(len(wave1), step=len(wave1)//100))
wave2 = f2(np.arange(len(wave2), step=len(wave2)//100))
plot_wave(wave1)
plot_wave(wave2)


#音をつなげて保存


def wavetoNsec(wave, sec, fs = 48000):
    tmp = wave.copy()
    for i in range((int)(fs*sec) // len(wave) - 1):
        wave = np.concatenate([wave, tmp])
    wave = np.concatenate([wave, tmp[0:(int)(fs*sec) % len(tmp)]])
    if (int)(fs*sec) < len(wave):
        return wave[:(int)(fs*sec)]
    return wave

wave1 = wavetoNsec(wave1, 0.5)
wave2 = wavetoNsec(wave2, 0.5)


#音量を上げる


wave1 = wave1 * 50.0
wave2 = wave2 * 50.0


#%%
#ピッチシフト用

sr = 48000

def printF0(wavdata, view = True):
    fmin, fmax = 5, 520
    fo_yin = librosa.yin(wavdata, fmin, fmax)
    F0 = np.mean(np.array(fo_yin))
    if view:
        print(F0)
    return F0

def shiftF0(wavdata, F0_shift, sr=48000):
    F0 = printF0(wavdata, view = False)
    n_steps = np.log2(F0_shift / F0)
    wav_shift = librosa.effects.pitch_shift(wavdata, sr, n_steps=n_steps, bins_per_octave=1)
    retF0 = printF0(wav_shift, view = False)
    return wav_shift, retF0

#周波数指定シフトが使い物にならないので二分探索でやる new:使い物になったので要らなくなった
def shiftF0_search(wavdata, F0_shift,  sr=48000, lb = 0.5, ub = 12.*9):
    cnt = 0
    while(ub - lb > 0.2 and cnt < 10):
        #cnt += 1
        mid = (lb + ub) / 2.0
        wav = librosa.effects.pitch_shift(wavdata, sr, n_steps=mid)
        thisF0 = printF0(wav, view=False)
        if(thisF0 < F0_shift):
            lb = mid
        else :
            ub = mid
    wav = librosa.effects.pitch_shift(wavdata, sr, n_steps=mid)
    F0 = printF0(wav, view=False)
    return wav, F0



#2つの音の高さをC3に合わせる


#wave1_C3, F0_wave1_C3 = shiftF0_search(wave1, 130.813)
wave1_C3, F0_wave1_C3 = shiftF0(wave1, 130.813)
print("wave1_C3.F0 = {}".format(F0_wave1_C3))

#wave2_C3, F0_wave2_C3 = shiftF0_search(wave2,  130.813)
wave2_C3, F0_wave2_C3 = shiftF0(wave2,  130.813)
print("wave2_C3.F0 = {}".format(F0_wave2_C3))


#C3でMFCCを比較してpianoが近い方をピアノにする


def calc_mfcc_diff(w1, w2, sr1, sr2, show=False):
    w1mf = librosa.feature.mfcc(y=w1, sr=sr1, n_mfcc=20, dct_type=3)
    w2mf = librosa.feature.mfcc(y=w2, sr=sr2, n_mfcc=20, dct_type=3)
    mean1 = np.mean(w1mf, 1)
    mean2 = np.mean(w2mf, 1)
    if show :
        plot_wave(mean1)
        plot_wave(mean2)
    return np.mean((mean1 - mean2)**2)

trumpet, srpet = librosa.load(librosa.ex('trumpet'))
trumpet = trumpet[0:4500] * 10000
#plot_wave(trumpet)
piano, srpiano = librosa.load("./outputs/piano.wav", sr=sr, mono=True)
piano = piano[2400:] * 10000
#plot_wave(piano)

piano_, tmpF0 = shiftF0(piano, 130.813, sr = srpiano)
dif1 = calc_mfcc_diff(wave1_C3, piano, sr, srpiano, show=False)
dif2 = calc_mfcc_diff(wave2_C3, piano, sr, srpiano, show=False)

if(dif1 > dif2):
    wave1, wave2 = wave2, wave1


#C3の音を基準にドドソソララソ(wave1)　ファファミミレレド(wave2)を作る


print("wave1_shift")
wave1_G3=librosa.effects.pitch_shift(wave1_C3, 48000, n_steps=7)
#wave1_G3, F0_wave1_G3 = shiftF0_search(wave1, 195.998)
printF0(wave1_G3)
wave1_A3=librosa.effects.pitch_shift(wave1_C3, 48000, n_steps=9)
#wave1_A3, F0_wave1_A3 = shiftF0_search(wave1, 220.000)
printF0(wave1_A3)


print("wave2_shift")
wave2_D3 = librosa.effects.pitch_shift(wave2_C3, 48000, n_steps=2)
#wave1_G3, F0_wave1_G3 = shiftF0_search(wave1, 195.998)
printF0(wave2_D3)
wave2_E3=librosa.effects.pitch_shift(wave2_C3, 48000, n_steps=4)
#wave1_A3, F0_wave1_A3 = shiftF0_search(wave1, 220.000)
printF0(wave2_E3)
wave2_F3=librosa.effects.pitch_shift(wave2_C3, 48000, n_steps=5)
#wave1_A3, F0_wave1_A3 = shiftF0_search(wave1, 220.000)
printF0(wave2_F3)

#from scipy import signal
#wave1_C3 = wave1_C3 * signal.hamming(len(wave1_C3))
#plot_wave(wave1_C3)
#wave1_G3 = wave1_G3 * signal.hamming(len(wave1_G3))
#wave1_A3 = wave1_A3 * signal.hamming(len(wave1_A3))
#print("mfcc")
#print(calc_mfcc_diff(wave1_C3, trumpet, sr, srpet, show=False))
#print(calc_mfcc_diff(wave2_C3, trumpet, sr, srpet, show=False))
#print(calc_mfcc_diff(wave1_C3, wave2_C3, sr, sr, show=False))



kuuhaku1 = np.zeros((6000))
kuuhaku2 = np.zeros((24000))



kirakiraboshi = np.concatenate([
    wave1_C3, 
    kuuhaku1, 
    wave1_C3, 
    kuuhaku1, 
    wave1_G3, 
    kuuhaku1, 
    wave1_G3,
    kuuhaku1, 
    wave1_A3,
    kuuhaku1, 
    wave1_A3,
    kuuhaku1, 
    wave1_G3,
    kuuhaku1,
    kuuhaku2,
    wave2_F3,
    kuuhaku1,
    wave2_F3,
    kuuhaku1,
    wave2_E3,
    kuuhaku1,
    wave2_E3,
    kuuhaku1,
    wave2_D3,
    kuuhaku1,
    wave2_D3,
    kuuhaku1,
    wave2_C3,
    kuuhaku1,
    ])
write_wave("./outputs/kirakiraboshi01.wav", kirakiraboshi.astype(np.int16))

#%%
#playsound("./outputs/kirakiraboshi01.wav")
#%%

#ピアノとトランペットで作った音楽を奏でる


piano = wavetoNsec(piano, 0.5, srpiano)
trumpet = wavetoNsec(trumpet, 1, srpet)

print("piano_shift")
piano_C3, F0_piano_C3 = shiftF0(piano, 130.813)
print(F0_piano_C3)
piano_G3=librosa.effects.pitch_shift(piano_C3, srpiano, n_steps=7)
printF0(piano_G3)
piano_A3=librosa.effects.pitch_shift(piano_C3, srpiano, n_steps=9)
printF0(piano_A3)
print("trumpet_shift")
trumpet_C3, F0_trumpet_C3 = shiftF0(trumpet,  130.813)
print(F0_trumpet_C3)
trumpet_D3 = librosa.effects.pitch_shift(trumpet_C3, srpet, n_steps=2)
printF0(trumpet_D3)
trumpet_E3=librosa.effects.pitch_shift(trumpet_C3, srpet, n_steps=4)
printF0(trumpet_E3)
trumpet_F3=librosa.effects.pitch_shift(trumpet_C3, srpet, n_steps=5)
printF0(trumpet_F3)





kirakiraboshi2 = np.concatenate([
    piano_C3, 
    kuuhaku1, 
    piano_C3, 
    kuuhaku1, 
    piano_G3, 
    kuuhaku1, 
    piano_G3,
    kuuhaku1, 
    piano_A3,
    kuuhaku1, 
    piano_A3,
    kuuhaku1, 
    piano_G3,
    kuuhaku1,
    kuuhaku2,

    wave1_C3, 
    kuuhaku1, 
    wave1_C3, 
    kuuhaku1, 
    wave1_G3, 
    kuuhaku1, 
    wave1_G3,
    kuuhaku1, 
    wave1_A3,
    kuuhaku1, 
    wave1_A3,
    kuuhaku1, 
    wave1_G3,
    kuuhaku1,
    kuuhaku2,

    trumpet_F3,
    kuuhaku1,
    trumpet_F3,
    kuuhaku1,
    trumpet_E3,
    kuuhaku1,
    trumpet_E3,
    kuuhaku1,
    trumpet_D3,
    kuuhaku1,
    trumpet_D3,
    kuuhaku1,
    trumpet_C3,
    kuuhaku1,
    kuuhaku2,

    wave2_F3,
    kuuhaku1,
    wave2_F3,
    kuuhaku1,
    wave2_E3,
    kuuhaku1,
    wave2_E3,
    kuuhaku1,
    wave2_D3,
    kuuhaku1,
    wave2_D3,
    kuuhaku1,
    wave2_C3,
    kuuhaku1,
    ])
write_wave("./outputs/kirakiraboshi02.wav", kirakiraboshi2.astype(np.int16))

def tmp_play(w1):
    write_wave("./outputs/tmp.wav", w1.astype(np.int16))
    playsound("./outputs/tmp.wav")

tmp_play(kirakiraboshi2)


#%%
'''
def autocorrelation_type1(sig):
    len_sig = len(sig)
    spec = np.fft.fft(sig)
    return np.fft.ifft(spec * spec.conj()).real[:len_sig]

def autocorrelation_type2(sig):
    len_sig = len(sig)
    sig = np.pad(sig, (0, len_sig), "constant")
    spec = np.fft.fft(sig)
    return np.fft.ifft(spec * spec.conj()).real[:len_sig]

def difference_type1(sig):
    autocorr = autocorrelation_type1(sig)
    return autocorr[0] - autocorr

def difference_type2(sig):
    autocorr = autocorrelation_type2(sig)
    energy = (sig * sig)[::-1].cumsum()[::-1]
    return energy[0] + energy - 2 * autocorr

def cumulative_mean_normalized_difference(diff):
    diff[0] = 1
    sum_value = 0
    for tau in range(1, len(diff)):
        sum_value += diff[tau]
        diff[tau] /= sum_value / tau
    return diff

sig = wave1_C3 # 任意の入力信号を取得。
diff = difference_type1(sig)
cmnd = cumulative_mean_normalized_difference(diff)

YIN_THRESHOLD = 0.3 # 任意の正の値のしきい値。

def absolute_threshold(diff, threshold=YIN_THRESHOLD):
    tau = 2
    while tau < len(diff):
        if diff[tau] < threshold:
            while tau + 1 < len(diff) and diff[tau + 1] < diff[tau]:
                tau += 1
            break
        tau += 1
    return None if tau == len(diff) or diff[tau] >= threshold else tau

def parabolic_interpolation(array, x):
    x_result = None
    if x < 1:
        x_result = x if array[x] <= array[x + 1] else x + 1
    elif x >= len(array) - 1:
        x_result = x if array[x] <= array[x - 1] else x - 1
    else:
        denom = array[x + 1] + array[x - 1] - 2 * array[x]
        delta = array[x - 1] - array[x + 1]
        if denom == 0:
            return x
        return x + delta / (2 * denom)
    return x_result

def yin_type1(sig, samplerate):
    diff = difference_type1(sig)
    cmnd = cumulative_mean_normalized_difference(diff)
    tau = absolute_threshold(cmnd)
    if tau is None:
        return np.nan
    return samplerate / parabolic_interpolation(cmnd, tau)

def yin_type2(sig, samplerate):
    diff = difference_type2(sig)
    cmnd = cumulative_mean_normalized_difference(diff)
    tau = absolute_threshold(cmnd)
    if tau is None:
        return np.nan
    return samplerate / parabolic_interpolation(cmnd, tau)


print(yin_type1(wave1[:100], 48000))
print(yin_type2(wave1[:100], 48000))

import pyworld as pw
plot_wave(wave1_C3[:]/50/400.)
y = (wave1_C3[:]/50/400.).astype(float)
_f0, _time = pw.dio(y, sr)
f0 = pw.stonemask(y, _f0, _time, sr)
print(f0)
#plt.plot(f0, linewidth=3, color="green", label="F0 contour")
#plt.legend(fontsize=10)
#plt.show()

'''
#%%

'''
tmp_play(wave1_C3)
'''
#%%
'''
def normalized_square_difference_type1(sig):
    corr = autocorrelation_type1(sig)
    return corr / corr[0] if corr[0] != 0 else corr

def normalized_square_difference_type2(sig):
    corr = autocorrelation_type2(sig)
    cumsum = (sig * sig)[::-1].cumsum()[::-1]
    cumsum[cumsum < 1] = 1  # 発散を防ぐ。
    return corr / (corr[0] + cumsum)

MPM_K = 0.5  # Type I NSD では後半に大きなピークができるので小さめに設定。

def estimate_period(diff):
    start = 0
    while diff[start] > 0:
        start += 1
        if start >= len(diff):
            return None

    threshold = MPM_K * np.max(diff[start:])
    isNegative = True
    max_index = None
    for i in range(start, len(diff)):
        if isNegative:
            if diff[i] < 0:
                continue
            max_index = i
            isNegative = False
        if diff[i] < 0:
            isNegative = True
            if diff[max_index] >= threshold:
                return max_index
        if diff[i] > diff[max_index]:
            max_index = i
    return None

def mpm_type1(data, samplerate):
    nsd = normalized_square_difference_type1(data)
    index = get_period(nsd)
    if index is None:
        return np.nan
    return samplerate / parabolic_interpolation(nsd, index)

def mpm_type2(data, samplerate):
    nsd = normalized_square_difference_type2(data)
    index = get_period(nsd)
    if index is None:
        return np.nan
    return samplerate / parabolic_interpolation(nsd, index)
'''
#%%

'''
#だめだった
fs = 48000
size = 4096
t = np.arange(0, size) / fs
han = np.hanning(size)

#ピークピッキング
def pick_peak(data):
    peaks_val = []
    peaks_index = []
    for i in range(2, data.size):
        if data[i-1] - data[i-2] >= 0 and data[i] - data[i-1] < 0:
            peaks_val.append(data[i-1])
            peaks_index.append(i-1)
    max_index = peaks_val.index(max(peaks_val))
    return peaks_index[max_index]


Y = np.fft.fft(wave1_C3[:4096]*han)
acf = np.fft.ifft(abs(Y)**2)
n = pick_peak(np.real(acf[0:size//2]))
f0 = fs / n
print("f0 = {}".format(f0))

printF0(wave1_C3[:4096])
'''
#%%
'''
def mado(w1):
    x = np.arange(len(w1))
    x = x * np.pi / len(w1)
    y = np.cos(x) * (np.pi-x) / 1.8
    return w1 * y

trumpet, srpet = librosa.load(librosa.ex('trumpet'))
trumpet = trumpet[0:4500]
piano, srpiano = librosa.load("./outputs/piano.wav", sr=sr, mono=True)
piano = piano[2500:]
plot_wave(wave1[:sr//10])
plot_wave(wave2[:sr//10])
plot_wave(trumpet[:srpet//10])
plot_wave(piano[:srpiano//10])

printF0(wave1)
printF0(wave2)
printF0(trumpet)
printF0(piano)
'''
#%%
'''
fmin, fmax = 5, 520
fo_yin = librosa.yin(wave1[:2000], fmin, fmax)
print(fo_yin.shape)
plot_wave(fo_yin)
print(fo_yin)
tmp_play(wave1_C3)
'''
#%%
'''
import sox
def pitch_shift_sox(data, sample_rate, shift):
    tfm = sox.Transformer()
    tfm.pitch(shift)
    return tfm.build_array(input_array=data, sample_rate_in=sample_rate)

data_pitch = pitch_shift_sox(wave1_C3, 48000, 12)
tmp_play(wave1_C3)
tmp_play(data_pitch)
'''
#%%
'''
write_wave("./tmp.wav", wave1_C3.astype(np.int16))
'''
#%%
'''
import sox
transformer = sox.Transformer()
transformer.pitch(n_semitones=12)
transformer.build("./outputs/tmp.wav", "./outputs/tmp2.wav")

playsound("./outputs/tmp.wav")
playsound("./outputs/tmp2.wav")
'''
#%%
'''
tmp, srtmp = librosa.load("./tmp2.wav", sr=sr, mono=True)
printF0(tmp)

'''
#%%
'''
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


pitch = F.detect_pitch_frequency(
    torch.Tensor(piano), 
    sample_rate=srpiano#, 
    #frame_time=0.5, 
    #win_length=30, 
    #freq_low=5, 
    #freq_high=400
)
print(pitch)
'''
#%%
'''
fmin, fmax = 5, 520
fo_yin = librosa.yin(wave1_C3, fmin, fmax)
print(fo_yin)
F0 = np.mean(np.array(fo_yin))

'''