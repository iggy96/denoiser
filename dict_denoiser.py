# import numpy file
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import DictionaryLearning

path = '/Users/joshuaighalo/Downloads/'
filename_eeg = 'EEG_all_epochs.npy'
data_eeg = np.load(path + filename_eeg)
cleanEEG_1D = data_eeg[0,:]
cleanEEG_1D = cleanEEG_1D.reshape(1, -1)
plt.plot(cleanEEG_1D.T)
plt.show()

# generate dictionary of atoms from data
dict = DictionaryLearning(n_components=10,alpha=1, max_iter=1000,random_state=42)
dict.fit(data_eeg)
dictionary = dict.components_

filename_emg = 'EMG_all_epochs.npy'
data_emg = np.load(path + filename_emg)
emg_1D = data_emg[0,:]
emg_1D = emg_1D.reshape(1, -1)
plt.plot(emg_1D.T)
plt.show()

filename_eog = 'EOG_all_epochs.npy'
data_eog = np.load(path + filename_eog)
eog_1D = data_eog[0,:]
eog_1D = eog_1D.reshape(1, -1)
plt.plot(eog_1D.T)
plt.show()

lamda = 1
eeg_emg = data_eeg[4000,:].reshape(1, -1) + lamda*emg_1D
eeg_eog = data_eeg[4000,:].reshape(1, -1) + lamda*eog_1D
plt.plot(eeg_emg.T)
plt.show()
plt.plot(eeg_eog.T)
plt.show()

def nmse(clean_signal, denoised_signal):
    """
    Calculates the Normalized Mean Squared Error (NMSE) between the clean signal and the denoised signal.
    clean_signal: np.array, the clean signal
    denoised_signal: np.array, the denoised signal
    """
    N = len(clean_signal)
    error = np.sum((clean_signal - denoised_signal)**2) / N
    variance = np.sum(clean_signal**2) / N
    nmse = error / variance
    print("NMSE: ", nmse)
    
    return nmse

def snr(x, x_denoised):
    """
    Calculate the signal-to-noise ratio (SNR) between the original clean EEG signal and the denoised EEG signal.

    Parameters
    ----------
    x : numpy.ndarray
        Clean EEG signal
    x_denoised : numpy.ndarray
        Denoised EEG signal

    Returns
    -------
    snr : float
        Signal-to-noise ratio

    """
    signal_power = np.sum(x**2) / len(x)
    noise_power = np.sum((x - x_denoised)**2) / len(x)
    snr = 10 * np.log10(signal_power / noise_power)
    print("SNR: ", snr)
    return snr


nmse(cleanEEG_1D, eeg_emg)
nmse(cleanEEG_1D, eeg_eog)
snr(cleanEEG_1D, eeg_emg)
snr(cleanEEG_1D, eeg_eog)

# apply the dictionary to the noisy signal
code = dict.transform(eeg_emg)
denoised_signal = np.dot(dictionary.T, code.T)
plt.plot(denoised_signal)
plt.show()

snr(cleanEEG_1D, denoised_signal.T)
nmse(cleanEEG_1D, denoised_signal.T)

