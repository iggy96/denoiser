import numpy as np
from scipy.signal import chirp, find_peaks
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

def generate_synthetic_clean_eeg(fs, duration):
    t = np.linspace(0, duration, int(fs*duration))
    f0, f1 = 1, 30
    signal = chirp(t, f0, t[-1], f1)
    noise = np.random.normal(0, 0.1, signal.shape)
    clean_eeg = signal + noise
    clean_eeg = (clean_eeg - np.min(clean_eeg)) / (np.max(clean_eeg) - np.min(clean_eeg))
    clean_eeg = clean_eeg * 100 - 50
    return clean_eeg

def calculate_snr(eeg_signal, fs):
    # Calculate the power spectral density of the signal
    freqs, psd = welch(eeg_signal, fs, nperseg=fs*2)
    # Find the indices of the EEG frequency range (0.5 - 100 Hz)
    eeg_indices = np.where((freqs >= 0.5) & (freqs <= 100))
    # Calculate the signal power in the EEG frequency range
    signal_power = np.sum(psd[eeg_indices])
    # Calculate the noise power in the EEG frequency range
    noise_power = np.sum(psd) - signal_power
    # Calculate the SNR
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

#%% clean eeg
fs, duration = 256, 60
clean_eeg = generate_synthetic_clean_eeg(fs, duration)
plt.plot(clean_eeg)
plt.show()
snr = calculate_snr(clean_eeg, fs)
print("SNR clean eeg: {:.2f} dB".format(snr))

#%% eye blink contaminated eeg
eeg = np.random.normal(0, 150, int(fs*duration))
blink_duration = 0.1 # in seconds
blink_amplitude = 600 # in microvolt
start_time = np.random.randint(0, int(fs*duration) - int(fs*blink_duration))
blink = np.zeros(int(fs*duration))
blink[start_time:start_time+int(fs*blink_duration)] = blink_amplitude
eeg_blink_contaminated = eeg + blink
plt.plot(eeg_blink_contaminated)
plt.show()
snr = calculate_snr(eeg_blink_contaminated, fs)
print("SNR (eye blink contaminated eeg): {:.2f} dB".format(snr))

#%% lateral eye movement contaminated eeg
fs = 256  # sampling frequency
n_channels = 4  # number of channels
n_samples = 2560  # number of samples
raw = np.random.randn(n_channels, n_samples)
frequency = 5  # frequency of the artifact
amplitude = 0.1  # amplitude of the artifact
t = np.arange(n_samples) / fs  # time vector
eye_movement = amplitude * np.sin(2*np.pi*frequency*t)
raw[0,:] += eye_movement
snr = 1  # desired SNR in dB
noise_power = np.var(raw) / 10**(snr / 10)  # calculate noise power
raw += np.random.normal(scale=np.sqrt(noise_power), size=raw.shape)
eeg_eyemovement = raw[0,:]
snr = calculate_snr(eeg_eyemovement, fs)
print("SNR (lateral eye movement contaminated eeg): {:.2f} dB".format(snr))
plt.plot(t, eeg_eyemovement)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

#%% chewing artifact contaminated eeg
fs = 256  # sampling frequency
n_channels = 4  # number of channels
n_samples = 2560  # number of samples
raw = np.random.randn(n_channels, n_samples)
f1, f2 = 20, 35  # frequency range of the artifact
amplitude = 0.1  # amplitude of the artifact
noise_power = 0.001  # power of the added noise
t = np.arange(n_samples) / fs  # time vector
chewing = amplitude * np.sin(2*np.pi*f1*t) + amplitude * np.sin(2*np.pi*f2*t) + np.random.normal(scale=np.sqrt(noise_power), size=n_samples)
frequency = 100  # frequency of the artifact
amplitude = 0.1  # amplitude of the artifact
hypoglossal = amplitude * np.sign(np.sin(2*np.pi*frequency*t))
raw[0,:] += chewing
raw[1,:] += hypoglossal
snr = calculate_snr(raw[0,:], fs)
print("SNR (chewing artifact contaminated eeg): {:.2f} dB".format(snr))
plt.plot(t, raw[0,:])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()


#%%
import emd


