import numpy as np
import matplotlib.pyplot as plt


fs = 1000
duration = 0.9
samples = int(fs * duration)
t = np.linspace(0, duration, samples, False)
amplitude_p100 = 3
amplitude_p200 = 3
amplitude_p300 = 5
amplitude_n100 = -1
amplitude_n200 = -8

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

# added alpha oscillations
def alpha_oscillation(t, amplitude_alpha, frequency_alpha):
    return amplitude_alpha * np.sin(2 * np.pi * frequency_alpha * t)


# Set the amplitude and frequency of the alpha oscillations
amplitude_alpha = 2
frequency_alpha = 10

# Generate the alpha oscillations
alpha = alpha_oscillation(t, amplitude_alpha, frequency_alpha)

# Add the alpha oscillations to the ERP waveform
y += alpha
contaminated_erp = y.reshape(len(y),1)

# Plot the ERP waveform with alpha oscillations
plt.plot(t, contaminated_erp)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.gca().invert_yaxis()
plt.title('Synthetic ERP Waveform with Alpha Oscillations')
plt.axvline(x=0, color='k', linestyle='--')
plt.xticks(np.arange(0, 0.9, 0.1))
plt.axhline(y=0, color='k', linestyle='-')
plt.show()
