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

#%%
# import npy file
path = "/Users/joshuaighalo/Downloads"
# load the npy file
eeg = np.load(path + "/EEG_all_epochs.npy")
eeg.shape
plt.plot(eeg[10,:])
plt.show()
#%% generate synthetic clean eeg
import numpy as np
import matplotlib.pyplot as plt

# Generate time axis (in seconds)
samp_freq = 500  # Sampling frequency (in Hz)
time = np.arange(0, 10, 1/samp_freq)  # 10 seconds of data

# Generate raw signal
raw_amp = np.random.uniform(10, 50, size=time.shape[0])  # Amplitude
raw_freq = np.random.uniform(0.5, 2, size=time.shape[0])  # Frequency
raw_signal = raw_amp * np.sin(2 * np.pi * raw_freq * time)

# Generate alpha rhythm
alpha_amp = np.random.uniform(10, 50, size=time.shape[0])  # Amplitude
alpha_freq = np.random.uniform(8, 12, size=time.shape[0])  # Frequency
alpha_signal = alpha_amp * np.sin(2 * np.pi * alpha_freq * time)

# Generate beta rhythm
beta_amp = np.random.uniform(10, 30, size=time.shape[0])  # Amplitude
beta_freq = np.random.uniform(13, 35, size=time.shape[0])  # Frequency
beta_signal = beta_amp * np.sin(2 * np.pi * beta_freq * time)

# Combine all signals to create the final EEG signal
eeg_signal = raw_signal + alpha_signal + beta_signal

# Plot the EEG signal
plt.plot(time, eeg_signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.title('Synthetic Clean EEG Signal')
plt.show()
print("SNR clean eeg: {:.2f} dB".format(calculate_snr(eeg_signal, samp_freq)))
# check quality of the synthetic clean eeg
freqs, psd = welch(eeg_signal, samp_freq, nperseg=samp_freq*2)
plt.plot(freqs, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2/Hz)')
plt.show()
#%% clean eeg
fs, duration = 500, 300
clean_eeg = generate_synthetic_clean_eeg(fs, duration)
ts = np.linspace(0, duration, int(fs*duration))
plt.plot(ts,clean_eeg)
plt.show()
print("SNR clean eeg: {:.2f} dB".format(calculate_snr(clean_eeg, fs)))
print("RMSE clean eeg: {:.2f} microvolt".format(np.sqrt(np.mean((clean_eeg - clean_eeg)**2))))

#%% generate 60Hz sinusoidal wave of fs=500Hz and duration=300s
fs, duration = 500, 300
ts = np.linspace(0, duration, int(fs*duration))
f0 = 60
line_noise = 1000 * np.sin(2*np.pi*f0*ts)
# merge line noise with clean eeg
eeg_line_noise_contaminated = clean_eeg + line_noise
plt.plot(ts,eeg_line_noise_contaminated)
plt.show()
print("SNR line noise: {:.2f} dB".format(calculate_snr(eeg_line_noise_contaminated, f0)))
print("RMSE line noise: {:.2f} microvolt".format(np.sqrt(np.mean((eeg_line_noise_contaminated - clean_eeg)**2))))
# check the frequency spectrum of the line noise contaminated eeg
freqs, psd = welch(eeg_line_noise_contaminated, fs, nperseg=fs*2)
plt.plot(freqs, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2/Hz)')
plt.show()

#%% generate random noise with fixed SNR and fixed seed
snr = 0.7  # desired SNR in dB
noise_power = np.var(clean_eeg) / 10**(snr / 10)  # calculate noise power
np.random.seed(42)
noise = 150*np.random.normal(scale=np.sqrt(noise_power), size=clean_eeg.shape)
def butterBandPass(data,lowcut,highcut,fs,order=4):
    """
        Inputs  :   data    - 2D numpy array (d0 = samples, d1 = channels) of unfiltered EEG data
                    low     - lower limit in Hz for the bandpass filter (defaults to config)
                    high    - upper limit in Hz for the bandpass filter (defaults to config)
                    fs      - sampling rate of hardware (defaults to config)
                    order   - the order of the filter (defaults to 4)  
        Output  :   y     - 2D numpy array (d0 = samples, d1 = channels) of notch-filtered EEG data
        NOTES   :   
        Todo    : report testing filter characteristics
        data: eeg data (samples, channels)
        some channels might be eog channels
    """
    from scipy.signal import sosfiltfilt, butter 
    low_n = lowcut
    high_n = highcut
    sos = butter(order, [low_n, high_n], btype="bandpass", analog=False, output="sos",fs=fs)
    y = sosfiltfilt(sos, data, axis=0)
    return y
ocular_artifact = butterBandPass(noise,1,3,fs,order=4)
eeg_ocular_artifact_contaminated = clean_eeg + ocular_artifact
plt.plot(ts,eeg_ocular_artifact_contaminated)
plt.show()
print("SNR ocular artifact: {:.2f} dB".format(calculate_snr(eeg_ocular_artifact_contaminated, fs)))
print("RMSE ocular artifact: {:.2f} microvolt".format(np.sqrt(np.mean((eeg_ocular_artifact_contaminated - clean_eeg)**2))))
#%% eye blink contaminated eeg
blink_duration = 0.1 # in seconds
blink_amplitude = 600 # in microvolt
start_time = np.random.randint(0, int(fs*duration) - int(fs*blink_duration))
blink = np.zeros(int(fs*duration))
for i in range(60):
    start_time = np.random.randint(0, int(fs*duration) - int(fs*blink_duration))
    blink[start_time:start_time+int(fs*blink_duration)] = blink_amplitude
eeg_blink_contaminated = clean_eeg + blink
snr = 0.7  # desired SNR in dB
noise_power = np.var(eeg_blink_contaminated) / 10**(snr / 10)  # calculate noise power
eeg_blink_contaminated += np.random.normal(scale=np.sqrt(noise_power), size=eeg_blink_contaminated.shape)
plt.plot(ts,eeg_blink_contaminated)
plt.show()
print("SNR (eye blink contaminated eeg): {:.2f} dB".format(calculate_snr(eeg_blink_contaminated, fs)))
print("RMSE (eye blink contaminated eeg): {:.2f} microvolt".format(np.sqrt(np.mean((eeg_blink_contaminated - clean_eeg)**2))))

#%% lateral eye movement contaminated eeg
frequency = 5  # frequency of the artifact
movement_duration = 0.5  # duration of the artifact
movement_amplitude = 200  # amplitude of the artifact
eye_movement = movement_amplitude * np.sin(2*np.pi*frequency*ts)
for i in range(10):
    start_time = np.random.randint(0, int(fs*duration) - int(fs*movement_duration))
    eye_movement[start_time:start_time+int(fs*movement_duration)] = movement_amplitude * np.sin(2*np.pi*frequency*ts[start_time:start_time+int(fs*movement_duration)])
eeg_eyemovement_contaminated = clean_eeg + eye_movement
snr = 0.7  # desired SNR in dB
noise_power = np.var(eeg_eyemovement_contaminated) / 10**(snr / 10)  # calculate noise power
eeg_eyemovement_contaminated += np.random.normal(scale=np.sqrt(noise_power), size=eeg_eyemovement_contaminated.shape)
plt.plot(ts, eeg_eyemovement_contaminated)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()
print("SNR (lateral eye movement contaminated eeg): {:.2f} dB".format(calculate_snr(eeg_eyemovement_contaminated, fs)))
print("RMSE (lateral eye movement contaminated eeg): {:.2f} microvolt".format(np.sqrt(np.mean((eeg_eyemovement_contaminated - clean_eeg)**2))))

#%%
import emd


