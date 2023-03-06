import numpy as np
import matplotlib.pyplot as plt

# Set the sampling rate
fs = 1000

# Set the duration of the wave
duration = 0.9

# Calculate the number of samples
samples = int(fs * duration)

# Create an array of time values
t = np.linspace(0, duration, samples, False)

# Set the amplitude of the waveforms
amplitude_p100 = 3
amplitude_p200 = 3
amplitude_p300 = 5
amplitude_n100 = -1
amplitude_n200 = -8

# Define the waveforms
def fast_p100(t, amplitude_p100):
    return amplitude_p100 * np.exp(-((t - (75e-3)) / (30e-3)) ** 2)

def fast_p200(t, amplitude_p200):
    return amplitude_p200 * np.exp(-((t - (220e-3)) / (50e-3)) ** 2)

def slow_p300(t, amplitude_p300):
    return amplitude_p300 * np.exp(-((t - (400e-3)) / (120e-3)) ** 2)

def fast_n100(t, amplitude_n100):
    return amplitude_n100 * np.exp(-((t - (120e-3)) / (150e-3)) ** 2)

def fast_n200(t, amplitude_n200):
    return amplitude_n200 * np.exp(-((t - (120e-3)) / (15e-3)) ** 2)


# Generate the ERP waveform
y = fast_p100(t, amplitude_p100) + fast_p200(t, amplitude_p200) + slow_p300(t, amplitude_p300) + fast_n100(t, amplitude_n100) + fast_n200(t, amplitude_n200)

# Plot the ERP waveform
plt.plot(t, y)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.gca().invert_yaxis()
plt.title('Synthetic ERP Waveform with Slow P300 and Fast N100')
plt.show()





#%% added alpha oscillations
def alpha_oscillation(t, amplitude_alpha, frequency_alpha):
    return amplitude_alpha * np.sin(2 * np.pi * frequency_alpha * t)


# Set the amplitude and frequency of the alpha oscillations
amplitude_alpha = 2
frequency_alpha = 10

# Generate the alpha oscillations
alpha = alpha_oscillation(t, amplitude_alpha, frequency_alpha)

# Add the alpha oscillations to the ERP waveform
y += alpha

# Plot the ERP waveform with alpha oscillations
plt.plot(t, y)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.gca().invert_yaxis()
plt.title('Synthetic ERP Waveform with Alpha Oscillations')
plt.show()


#%%
# use EMD to decompose the signal into its components
import emd
imfs = emd.sift.mask_sift(y)
emd.plotting.plot_imfs(imfs[:fs*3, :])
plt.gca().invert_yaxis()
plt.show()

next_imfs = emd.sift.mask_sift(imfs[:,0])
emd.plotting.plot_imfs(next_imfs[:fs*3, :])
plt.gca().invert_yaxis()
plt.show()
