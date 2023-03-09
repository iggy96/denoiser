from template_erp import*
from noisy_erp import*
import numpy as np
from scipy.signal import correlate



def rolling_window(data_array,timing_array,window_size,step_size):
    """
    Inputs:
    1. data_array - 1D numpy array (d0 = channels) of data
    2. timing_array - 1D numpy array (d0 = samples) of timing data
    3. len(data_array) == len(timing_array)
    4. window_size - number of samples to use in each window in seconds e.g. 1 is 1 second
    5. step_size - the step size in seconds e.g.0.5 is 0.5 seconds 

    Outputs:    
    1. data_windows - 2D numpy array (d0 = windows, d1 = window size) of data

    """
    idx_winSize = np.where(timing_array == window_size)[0][0]
    idx_stepSize = np.where(timing_array == step_size)[0][0]
    shape = (data_array.shape[0] - idx_winSize + 1, idx_winSize)
    strides = (data_array.strides[0],) + data_array.strides
    rolled = np.lib.stride_tricks.as_strided(data_array, shape=shape, strides=strides)
    return rolled[np.arange(0,shape[0],idx_stepSize)]

def normalized_cross_correlation(template, data):
    # Normalize the template waveform to a unit range
    template_norm = template / np.abs(template).max()

    # Compute the mean and standard deviation of the input data
    data_mean = np.mean(data)
    data_std = np.std(data)

    # Normalize the input data to zero mean and unit standard deviation
    data_norm = (data - data_mean) / data_std

    # Compute the cross-correlation between the template and input data
    corr = correlate(data_norm, template_norm, mode='same')

    # Compute the normalized cross-correlation between the template and input data
    norm_corr = corr / (len(template) * data_std * np.abs(template).sum())

    # Return the normalized cross-correlation function
    return norm_corr


segments = rolling_window(contaminated_erp,t,0.2,0.05)


ncc_N1,simscore_N1 = [],[]
for i in range(segments.shape[0]):
    ncc_N1.append(normalized_cross_correlation(N1,segments[i]))
    simscore_N1.append(np.amax(ncc_N1[i])/1)
ncc_N1 = np.array(ncc_N1)
sim_score_N1 = np.array(simscore_N1)

max_idx = np.argmax(simscore_N1)
min_idx = np.argmin(simscore_N1)

# plot the row segments with each of their similarity scores
plt.figure(figsize=(10, 5))
for i in range(segments.shape[0]):
    plt.plot(segments[i], alpha=0.5, label='sim score: {}'.format(simscore_N1[i]))
plt.legend()
plt.gca().invert_yaxis()
plt.show()

#
ncc_P3,simscore_P3 = [],[]
for i in range(segments.shape[0]):
    ncc_P3.append(normalized_cross_correlation(P3,segments[i]))
    simscore_P3.append(np.amax(ncc_P3[i])/1)
ncc_P3 = np.array(ncc_P3)
sim_score_P3 = np.array(simscore_P3)

max_idx = np.argmax(simscore_P3)
min_idx = np.argmin(simscore_P3)

# plot the row segments with each of their similarity scores
plt.figure(figsize=(10, 5))
for i in range(segments.shape[0]):
    plt.plot(segments[i], alpha=0.5, label='sim score: {}'.format(simscore_P3[i]))
plt.legend()
plt.gca().invert_yaxis()
plt.show()


plt.plot(segments[7])
plt.gca().invert_yaxis()
plt.show()