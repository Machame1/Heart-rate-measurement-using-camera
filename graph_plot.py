import os
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import scipy.fftpack
from scipy.signal import butter, lfilter

# Define the folder and file path
folder_path = os.path.dirname(__file__)  # Current folder where the script is located
file_path = os.path.join(folder_path, 'signal.dat')

arr_red = []
arr_green = []
arr_blue = []

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y  

try:
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            r, g, b = line.strip().split("%")
            arr_red.append(float(r))
            arr_green.append(float(g))
            arr_blue.append(float(b))

except FileNotFoundError:
    print(f"File not found: {file_path}")
    raise

green_detrended = signal.detrend(arr_blue)
L = len(arr_red)

bpf = butter_bandpass_filter(green_detrended, 0.8, 3, fs=30, order=3)

even_times = np.linspace(0, L, L)
interpolated = np.interp(even_times, even_times, bpf)
interpolated = np.hamming(L) * interpolated
norm = interpolated / np.linalg.norm(interpolated)
raw = np.fft.rfft(norm * 30)
freq = np.fft.rfftfreq(L, 1 / 30) * 60
fft = np.abs(raw) ** 2

g = plt.figure("green")
ax2 = g.add_subplot(111)    
ax2.set_title("band pass filter")
ax2.set_xlabel("time")
ax2.set_ylabel("magnitude")
plt.plot(freq, fft, color="blue")
g.show()

input("Press Enter to exit...")
