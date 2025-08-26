import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, iirnotch, filtfilt
from sklearn.decomposition import FastICA

# --- Signal Parameters ---
Fs = 256  # Sampling rate in Hz
T = 10    # Duration in seconds
N = int(Fs * T) # Number of samples
t = np.linspace(0, T, N, endpoint=False) # Time vector

# --- 1. Simulate Clean EEG Signals for Multiple Channels ---
# Simplified brain activity (alpha wave)
clean_eeg_ch1 = 2 * np.sin(2 * np.pi * 10 * t)
clean_eeg_ch2 = 1.8 * np.sin(2 * np.pi * 11 * t + np.pi/4) # Slightly different alpha
clean_eeg_ch3 = 1.5 * np.sin(2 * np.pi * 9 * t + np.pi/2)  # Slightly different alpha

# --- 2. Simulate Common Artifacts for Multi-Channel Data ---

# a) Eye Blink Artifact (Physiological - EOG) - Stronger on Channel 1 (frontal)
# Large, slow deflection
blink_artifact = np.zeros(N)
blink_start_idx = int(2.0 * Fs)
blink_end_idx = int(3.0 * Fs)
blink_amplitude = 20 # Larger amplitude for prominent artifact
blink_duration_samples = blink_end_idx - blink_start_idx
blink_shape = blink_amplitude * (1 - np.cos(np.linspace(0, 2 * np.pi, blink_duration_samples))) / 2
blink_artifact[blink_start_idx:blink_end_idx] = blink_shape

# b) Muscle Activity (Physiological - EMG) - Stronger on Channel 2 (temporal)
# High-frequency noise
emg_noise = 3 * np.random.rand(N) * np.sin(2 * np.pi * np.linspace(30, 80, N) * t)

# c) Power Line Noise (Non-Physiological) - Common to all channels
power_line_freq = 50 # Hz (e.g., in Bangladesh)
power_line_noise_common = 5 * np.sin(2 * np.pi * power_line_freq * t + np.random.rand() * np.pi)

# --- 3. Create Raw Multi-Channel EEG Signal ---
raw_multi_channel_eeg = np.zeros((N, 3)) # N samples, 3 channels

raw_multi_channel_eeg[:, 0] = clean_eeg_ch1 + blink_artifact + power_line_noise_common + 0.5 * np.random.randn(N)
raw_multi_channel_eeg[:, 1] = clean_eeg_ch2 + emg_noise + power_line_noise_common + 0.5 * np.random.randn(N)
raw_multi_channel_eeg[:, 2] = clean_eeg_ch3 + power_line_noise_common + 0.5 * np.random.randn(N)


# --- 4. Preprocessing for ICA (Optional but Recommended) ---
# ICA often performs better if data is bandpass filtered and centered (zero mean).
# Here, we'll apply a common bandpass filter to remove very slow drifts and very high freq noise
# that ICA might struggle with or might not be of interest for brain activity.
nyquist_freq = 0.5 * Fs
b_bandpass, a_bandpass = butter(4, [1/nyquist_freq, 40/nyquist_freq], btype='band') # 1-40 Hz bandpass
preprocessed_eeg = filtfilt(b_bandpass, a_bandpass, raw_multi_channel_eeg, axis=0)
print(f"b_bandpass")