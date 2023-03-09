"""
Tmplate ERP waveform based on the following paper:
https://www.researchgate.net/publication/45667042_Neural_correlates_of_target_detection_in_the_attentional_blink
Amplitude is arbitrary and is not based on any empirical data.
latency is based off paper
"""
import numpy as np
import matplotlib.pyplot as plt


fs = 256
duration = 0.9
samples = int(fs * duration)
t = np.linspace(-0.1, duration, samples, False)
amplitude_p100 = 2
amplitude_p200 = 3
amplitude_p300 = 5
amplitude_n100 = -0.1
amplitude_n200 = -4

# Define the waveforms
def fast_p100(t, amplitude_p100):
    return amplitude_p100 * np.exp(-((t - (55e-3)) / (30e-3)) ** 2)

def fast_p200(t, amplitude_p200):
    return amplitude_p200 * np.exp(-((t - (220e-3)) / (50e-3)) ** 2)

def slow_p300(t, amplitude_p300):
    return amplitude_p300 * np.exp(-((t - (400e-3)) / (120e-3)) ** 2)

def fast_n100(t, amplitude_n100):
    return amplitude_n100 * np.exp(-((t - (120e-3)) / (150e-3)) ** 2)

def fast_n200(t, amplitude_n200):
    return amplitude_n200 * np.exp(-((t - (120e-3)) / (15e-3)) ** 2)


# Generate the ERP waveform
template_erp = fast_p100(t, amplitude_p100) + fast_p200(t, amplitude_p200) + slow_p300(t, amplitude_p300) + fast_n100(t, amplitude_n100) + fast_n200(t, amplitude_n200)
template_erp = template_erp.reshape(len(template_erp),1)

# Plot the ERP waveform
plt.plot(t,template_erp)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.gca().invert_yaxis()
plt.title('Template ERP Waveform with Slow P300 and Fast N100')
plt.axvline(x=0, color='k', linestyle='--')
plt.xticks(np.arange(0, 0.9, 0.1))
plt.axhline(y=0, color='k', linestyle='-')
plt.show()

curveN1 = template_erp[44:57]
curveP3 = template_erp[89:172]
peakN1 = np.amin(template_erp)
peakP3 = np.amax(template_erp)
idx_peakN1 = np.where(template_erp==peakN1)[0]
idx_peakP3 = np.where(template_erp==peakP3)[0]
print("template ERP N1 amplitude:",np.amin(template_erp))
print("template ERP N1 index:",np.where(template_erp==peakN1)[0])
print("template ERP N1 latency:",t[np.where(template_erp==peakN1)[0]])
print("template ERP P3 amplitude:",np.amax(template_erp))
print("template ERP P3 index:",np.where(template_erp==peakP3)[0])
print("template ERP P3 latency:",t[np.where(template_erp==peakP3)[0]])