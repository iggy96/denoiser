import bisect
import numpy as np
import pandas as pd
import pywt
from scipy import stats, signal, integrate
from dit.other import tsallis_entropy
import dit
import librosa
from scipy import signal
import statsmodels.api as sm
import itertools
from pyinform import mutualinfo
from statsmodels import tsa
from sklearn.metrics import mutual_info_score
import numpy as np
from scipy import signal,integrate
from sklearn.metrics.cluster import normalized_mutual_info_score as normed_mutual_info 
from pyentrp import entropy as ent
import antropy as ant
import warnings
from mne.time_frequency import psd_array_multitaper
from scipy.integrate import simps
warnings.filterwarnings("ignore", category=RuntimeWarning) 



################################################
#	Auxiliary Functions
################################################

##########
# Filter the eegData, midpass filter 
#	eegData: 3D np array [chans x ms x epochs] 
def filt_data(eegData, lowcut, highcut, fs, order=7):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filt_eegData = signal.lfilter(b, a, eegData, axis = 1)
    return filt_eegData
 

def resample_eeg(eeg_data, old_sampling_rate, new_sampling_rate):
    """
    Resamples EEG data from an old sampling rate to a new sampling rate.

    Args:
    - eeg_data (ndarray): EEG data array with shape (num_channels, num_samples)
    - old_sampling_rate (float): the original sampling rate (in Hz)
    - new_sampling_rate (float): the desired sampling rate (in Hz)

    Returns:
    - resampled_data (ndarray): resampled EEG data array with shape (num_channels, num_samples_new)
    """

    # Calculate the resampling factor
    resampling_factor = new_sampling_rate / old_sampling_rate

    # Use the scipy.signal.resample function to resample the data
    num_channels, num_samples = eeg_data.shape
    num_samples_new = int(np.ceil(num_samples * resampling_factor))
    time_old = np.arange(num_samples) / old_sampling_rate
    time_new = np.arange(num_samples_new) / new_sampling_rate
    resampled_data = np.zeros((num_channels, num_samples_new))
    for channel in range(num_channels):
        resampled_data[channel, :] = signal.resample(eeg_data[channel, :], num_samples_new)

    return resampled_data

##########
# Extract the Shannon Entropy
# threshold the signal and make it discrete, normalize it and then compute entropy
def shannonEntropy(eegData, bin_min, bin_max, binWidth):
    H = np.zeros((eegData.shape[0], eegData.shape[2]))
    for chan in range(H.shape[0]):
        for epoch in range(H.shape[1]):
            counts, binCenters = np.histogram(eegData[chan,:,epoch], bins=np.arange(bin_min+1, bin_max, binWidth))
            nz = counts > 0
            prob = counts[nz] / np.sum(counts[nz])
            H[chan, epoch] = -np.dot(prob, np.log2(prob/binWidth))
            #H = H.reshape(H.shape[1],)
    return H.reshape(H.shape[1],)
    

##########
# Lyapunov exponent
def lyapunov(eegData):
    l = np.mean(np.log(np.abs(np.gradient(eegData,axis=1))),axis=1)
    return l.reshape(l.shape[1],)
    
##########
# Fractal Embedding Dimension
# From pyrem: packadge for sleep scoring from EEG data
# https://github.com/gilestrolab/pyrem/blob/master/src/pyrem/univariate.py

def hFD(a, k_max): 
    def paramshFD(a, k_max): #Higuchi FD
        L = []
        x = []
        N = len(a)

        for k in range(1,k_max):
            Lk = 0
            for m in range(0,k):
                #we pregenerate all idxs
                idxs = np.arange(1,int(np.floor((N-m)/k)),dtype=np.int32)
                Lmk = np.sum(np.abs(a[m+idxs*k] - a[m+k*(idxs-1)]))
                Lmk = (Lmk*(N - 1)/(((N - m)/ k)* k)) / k
                Lk += Lmk

            L.append(np.log(Lk/(m+1)))
            x.append([np.log(1.0/ k), 1])

        (p, r1, r2, s)=np.linalg.lstsq(x, L)
        return p[0]
    
    output = []
    for i in range(a.shape[2]):
        output.append(paramshFD(a[0,:,i],k_max))
    output = np.array(output)
    return output


##########
# Hjorth Mobility
# Hjorth Complexity
# variance = mean(signal^2) iff mean(signal)=0
# which it is be because I normalized the signal
# Assuming signals have mean 0
# Mobility = sqrt( mean(dx^2) / mean(x^2) )
def hjorthParameters(xV):
    dxV = np.diff(xV, axis=1)
    ddxV = np.diff(dxV, axis=1)

    mx2 = np.mean(np.square(xV), axis=1)
    mdx2 = np.mean(np.square(dxV), axis=1)
    mddx2 = np.mean(np.square(ddxV), axis=1)

    mob = mdx2 / mx2
    complexity = np.sqrt((mddx2 / mdx2) / mob)
    mobility = np.sqrt(mob)
    complexity = complexity.reshape(complexity.shape[1],)
    mobility = mobility.reshape(mobility.shape[1],)

    # PLEASE NOTE that Mohammad did NOT ACTUALLY use hjorth complexity,
    # in the matlab code for hjorth complexity subtraction by mob not division was used 
    return mobility, complexity


# median frequency
def medianFreq(eegData,fs):
    H = np.zeros((eegData.shape[0], eegData.shape[2]))
    for chan in range(H.shape[0]):
        freqs, powers = signal.periodogram(eegData[chan, :, :], fs, axis=0)
        H[chan,:] = freqs[np.argsort(powers,axis=0)[len(powers)//2]]
        #H = H.reshape(H.shape[1],)
    return H.reshape(H.shape[1],)


###########
# Regularity (burst-suppression)
# Regularity of eeg
# filter with a window of 0.5 seconds to create a nonnegative smooth signal.
# In this technique, we first squared the signal and applied a moving-average
# The window length of the moving average was set at 0.5 seconds.
def eegRegularity(eegData, Fs=100):
    in_x = np.square(eegData)  # square signal
    num_wts = Fs//2  # find the filter length in samples - we want 0.5 seconds.
    q = signal.lfilter(np.ones(num_wts) / num_wts, 1, in_x, axis=1)
    q = -np.sort(-q, axis=1) # descending sort on smooth signal
    N = q.shape[1]
    u2 = np.square(np.arange(1, N+1))
    # COMPUTE THE Regularity
    # dot each 5min epoch with the quadratic data points and then normalize by the size of the dotted things    
    reg = np.sqrt( np.einsum('ijk,j->ik', q, u2) / (np.sum(q, axis=1)*(N**2)/3) )
    reg = reg.reshape(reg.shape[1],)
    return reg

###########
# Voltage < (5μ, 10μ, 20μ)
def eegVoltage(eegData,voltage=20):
    eegFilt = eegData.copy()
    eegFilt[abs(eegFilt) > voltage] = np.nan
    volt_res = np.nanmean(eegFilt,axis=1)
    volt_res = volt_res.reshape(volt_res.shape[1],)
    return volt_res


## Connectivity Features
def sample_entropy(x):
    def params_sample_entropy(y):
        return ant.sample_entropy(y)
    
    output = []
    for i in range(x.shape[1]):
        output.append(params_sample_entropy(x[:,i]))
    output = np.array(output).T
    return output

def multiscale_entropy(x):
    def params_multiscale_entropy(y):
        return ent.multiscale_entropy(y, 1, 0.1 * np.std(y),1)
    
    output = []
    for i in range(x.shape[1]):
        output.append(params_multiscale_entropy(x[:,i]))
    output = np.array(output).T
    return output.reshape(output.shape[1],)

def permutation_entropy(x):
    def params_permutation_entropy(y):
        return ant.perm_entropy(y, normalize=True)
    
    output = []
    for i in range(x.shape[1]):
        output.append(params_permutation_entropy(x[:,i]))
    output = np.array(output).T
    return output

def spectral_entropy(x,fs):
    def params_spectral_entropy(y,fs):
        return ant.spectral_entropy(y, fs, method='welch', normalize=True)
    
    output = []
    for i in range(x.shape[1]):
        output.append(params_spectral_entropy(x[:,i],fs))
    output = np.array(output).T
    return output

def svd_entropy(x):
    def params_svd_entropy(y):
        return ant.svd_entropy(y, normalize=True)
    
    output = []
    for i in range(x.shape[1]):
        output.append(params_svd_entropy(x[:,i]))
    output = np.array(output).T
    return output

def app_entropy(x):
    def params_app_entropy(y):
        return ant.app_entropy(y)
    
    output = []
    for i in range(x.shape[1]):
        output.append(params_app_entropy(x[:,i]))
    output = np.array(output).T
    return output

def lziv(x):
    def params_lziv(y):
        return ant.lziv_complexity(y)
    
    output = []
    for i in range(x.shape[1]):
        output.append(params_lziv(x[:,i]))
    output = np.array(output).T
    return output

def petrosian(x):
    def params_petrosian(y):
        return ant.petrosian_fd(y)
    
    output = []
    for i in range(x.shape[1]):
        output.append(params_petrosian(x[:,i]))
    output = np.array(output).T
    return output

def katz(x):
    def params_katz(y):
        return ant.katz_fd(y)
    
    output = []
    for i in range(x.shape[1]):
        output.append(params_katz(x[:,i]))
    output = np.array(output).T
    return output

def dfa(x):
    def params_dfa(y):
        return ant.detrended_fluctuation(y)
    
    output = []
    for i in range(x.shape[1]):
        output.append(params_dfa(x[:,i]))
    output = np.array(output).T
    return output

def relative_bandpower(data_,bandFreq,sampFreq):
    def params_relbandpower(data, sf, band):
        """Compute the average power of the signal x in a specific frequency band.

        Requires MNE-Python >= 0.14.

        Parameters
        ----------
        data : 1d-array
        Input signal in the time-domain.
        sf : float
        Sampling frequency of the data.
        band : list
        Lower and upper frequencies of the band of interest.
        method : string
        Periodogram method: 'welch' or 'multitaper'
        window_sec : float
        Length of each window in seconds. Useful only if method == 'welch'.
        If None, window_sec = (1 / min(band)) * 2.
        relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

        Return
        ------
        bp : float
        Absolute or relative band power.
        """

        band = np.asarray(band)
        low, high = band
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                                normalization='full', verbose=30)
        freq_res = freqs[1] - freqs[0]
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        bp = simps(psd[idx_band], dx=freq_res)
        bp /= simps(psd, dx=freq_res)
        return bp
    
    output = []
    for i in range(data_.shape[1]):
        output.append(params_relbandpower(data_[:,i],sampFreq,bandFreq))
    return np.array(output).T


# Standard Deviation
def eegStd(eegData):
    std_res = np.std(eegData,axis=1)
    std_res = std_res.reshape(std_res.shape[1],)
    return std_res



def ratioDeltaTheta(eegData,fs):
    # calculate the power
    powers_delta = relative_bandpower(eegData, [0.5, 4], fs)
    powers_theta = relative_bandpower(eegData, [4, 8], fs)
    ratio_res = powers_delta / powers_theta
    return ratio_res

def ratioThetaDelta(eegData,fs):
    # calculate the power
    powers_delta = relative_bandpower(eegData, [0.5, 4], fs)
    powers_theta = relative_bandpower(eegData, [4, 8], fs)
    ratio_res = powers_theta / powers_delta
    return ratio_res

def ratioDeltaAlpha(eegData,fs):
    # calculate the power
    powers_delta = relative_bandpower(eegData, [0.5, 4], fs)
    powers_alpha = relative_bandpower(eegData, [8, 12], fs)
    ratio_res = powers_delta / powers_alpha
    return ratio_res

def ratioAlphaDelta(eegData,fs):
    # calculate the power
    powers_delta = relative_bandpower(eegData, [0.5, 4], fs)
    powers_alpha = relative_bandpower(eegData, [8, 12], fs)
    ratio_res = powers_alpha / powers_delta
    return ratio_res

def ratioDeltaBeta(eegData,fs):
    # calculate the power
    powers_delta = relative_bandpower(eegData, [0.5, 4], fs)
    powers_beta = relative_bandpower(eegData, [12, 30], fs)
    ratio_res = powers_delta / powers_beta
    return ratio_res

def ratioBetaDelta(eegData,fs):
    # calculate the power
    powers_delta = relative_bandpower(eegData, [0.5, 4], fs)
    powers_beta = relative_bandpower(eegData, [12, 30], fs)
    ratio_res = powers_beta / powers_delta
    return ratio_res

def ratioDeltaGamma(eegData,fs):
    # calculate the power
    powers_delta = relative_bandpower(eegData, [0.5, 4], fs)
    powers_gamma = relative_bandpower(eegData, [30, 45], fs)
    ratio_res = powers_delta / powers_gamma
    return ratio_res

def ratioGammaDelta(eegData,fs):
    # calculate the power
    powers_delta = relative_bandpower(eegData, [0.5, 4], fs)
    powers_gamma = relative_bandpower(eegData, [30, 45], fs)
    ratio_res = powers_gamma / powers_delta
    return ratio_res

def ratioThetaAlpha(eegData,fs):
    # calculate the power
    powers_theta = relative_bandpower(eegData, [4, 8], fs)
    powers_alpha = relative_bandpower(eegData, [8, 12], fs)
    ratio_res = powers_theta / powers_alpha
    return ratio_res

def ratioAlphaTheta(eegData,fs):
    # calculate the power
    powers_theta = relative_bandpower(eegData, [4, 8], fs)
    powers_alpha = relative_bandpower(eegData, [8, 12], fs)
    ratio_res = powers_alpha / powers_theta
    return ratio_res

def ratioThetaBeta(eegData,fs):
    # calculate the power
    powers_theta = relative_bandpower(eegData, [4, 8], fs)
    powers_beta = relative_bandpower(eegData, [12, 30], fs)
    ratio_res = powers_theta / powers_beta
    return ratio_res

def ratioBetaTheta(eegData,fs):
    # calculate the power
    powers_theta = relative_bandpower(eegData, [4, 8], fs)
    powers_beta = relative_bandpower(eegData, [12, 30], fs)
    ratio_res = powers_beta / powers_theta
    return ratio_res

def ratioThetaGamma(eegData,fs):
    # calculate the power
    powers_theta = relative_bandpower(eegData, [4, 8], fs)
    powers_gamma = relative_bandpower(eegData, [30, 45], fs)
    ratio_res = powers_theta / powers_gamma
    return ratio_res

def ratioGammaTheta(eegData,fs):
    # calculate the power
    powers_theta = relative_bandpower(eegData, [4, 8], fs)
    powers_gamma = relative_bandpower(eegData, [30, 45], fs)
    ratio_res = powers_gamma / powers_theta
    return ratio_res

def ratioAlphaBeta(eegData,fs):
    # calculate the power
    powers_alpha = relative_bandpower(eegData, [8, 12], fs)
    powers_beta = relative_bandpower(eegData, [12, 30], fs)
    ratio_res = powers_alpha / powers_beta
    return ratio_res

def ratioBetaAlpha(eegData,fs):
    # calculate the power
    powers_alpha = relative_bandpower(eegData, [8, 12], fs)
    powers_beta = relative_bandpower(eegData, [12, 30], fs)
    ratio_res = powers_beta / powers_alpha
    return ratio_res

def ratioAlphaGamma(eegData,fs):
    # calculate the power
    powers_alpha = relative_bandpower(eegData, [8, 12], fs)
    powers_gamma = relative_bandpower(eegData, [30, 45], fs)
    ratio_res = powers_alpha / powers_gamma
    return ratio_res

def ratioGammaAlpha(eegData,fs):
    # calculate the power
    powers_alpha = relative_bandpower(eegData, [8, 12], fs)
    powers_gamma = relative_bandpower(eegData, [30, 45], fs)
    ratio_res = powers_gamma / powers_alpha
    return ratio_res

def ratioBetaGamma(eegData,fs):
    # calculate the power
    powers_beta = relative_bandpower(eegData, [12, 30], fs)
    powers_gamma = relative_bandpower(eegData, [30, 45], fs)
    ratio_res = powers_beta / powers_gamma
    return ratio_res

def ratioGammaBeta(eegData,fs):
    # calculate the power
    powers_beta = relative_bandpower(eegData, [12, 30], fs)
    powers_gamma = relative_bandpower(eegData, [30, 45], fs)
    ratio_res = powers_gamma / powers_beta
    return ratio_res


def shortSpikeNum(eegData,minNumSamples=7,stdAway = 3):
    H = np.zeros((eegData.shape[0], eegData.shape[2]))
    for chan in range(H.shape[0]):
        for epoch in range(H.shape[1]):
            mean = np.mean(eegData[chan, :, epoch])
            std = np.std(eegData[chan,:,epoch])
            longSpikes = set(signal.find_peaks(abs(eegData[chan,:,epoch]-mean), 3*std,epoch,width=7)[0])
            shortSpikes = set(signal.find_peaks(abs(eegData[chan,:,epoch]-mean), 3*std,epoch,width=1)[0])
            H[chan,epoch] = len(shortSpikes.difference(longSpikes))
            #H = H.reshape((H.shape[1],))
    return H.reshape((H.shape[1],))




def compile_features(data,fs,delta,theta,alpha,beta,gamma,data_name,label):
    """
    data: (1, epoch_length, num_epochs)
    """
    delta_data = filt_data(data, delta[0], delta[1], fs)
    theta_data = filt_data(data, theta[0], theta[1], fs)
    alpha_data = filt_data(data, alpha[0], alpha[1], fs)
    beta_data = filt_data(data, beta[0], beta[1], fs)
    gamma_data = filt_data(data, gamma[0], gamma[1], fs)

    # Time Domain Features
    print("Calculating time domain features...")
    shannonRes = shannonEntropy(data, bin_min=-200, bin_max=200, binWidth=2)
    LyapunovRes = lyapunov(data)
    HiguchiFD_Res  = hFD(data,3)
    HjorthMob, HjorthComp = hjorthParameters(data)
    medianFreqRes = medianFreq(data, fs)
    delta_rbandpwr = relative_bandpower(data[0], delta, fs)
    theta_rbandpwr = relative_bandpower(data[0], theta, fs)
    alpha_rbandpwr = relative_bandpower(data[0], alpha, fs)
    beta_rbandpwr = relative_bandpower(data[0], beta, fs)
    gamma_rbandpwr = relative_bandpower(data[0], gamma, fs)
    stdRes = eegStd(data)
    delta_theta_ratio = ratioDeltaTheta(data[0],fs)
    theta_delta_ratio = ratioThetaDelta(data[0],fs)
    delta_alpha_ratio = ratioDeltaAlpha(data[0],fs)
    alpha_delta_ratio = ratioAlphaDelta(data[0],fs)
    delta_beta_ratio = ratioDeltaBeta(data[0],fs)
    beta_delta_ratio = ratioBetaDelta(data[0],fs)
    delta_gamma_ratio = ratioDeltaGamma(data[0],fs)
    gamma_delta_ratio = ratioGammaDelta(data[0],fs)
    theta_alpha_ratio = ratioThetaAlpha(data[0],fs)
    alpha_theta_ratio = ratioAlphaTheta(data[0],fs)
    theta_beta_ratio = ratioThetaBeta(data[0],fs)
    beta_theta_ratio = ratioBetaTheta(data[0],fs)
    theta_gamma_ratio = ratioThetaGamma(data[0],fs)
    gamma_theta_ratio = ratioGammaTheta(data[0],fs)
    alpha_beta_ratio = ratioAlphaBeta(data[0],fs)
    beta_alpha_ratio = ratioBetaAlpha(data[0],fs)
    alpha_gamma_ratio = ratioAlphaGamma(data[0],fs)
    gamma_alpha_ratio = ratioGammaAlpha(data[0],fs)
    beta_gamma_ratio = ratioBetaGamma(data[0],fs)
    gamma_beta_ratio = ratioGammaBeta(data[0],fs)
    regularity_res = eegRegularity(data,fs)
    volt05_res = eegVoltage(data,5)
    volt10_res = eegVoltage(data,10)
    volt20_res = eegVoltage(data,20)
    volt50_res = eegVoltage(data,50)
    volt100_res = eegVoltage(data,100)
    minNumSamples = int(70*fs/1000)
    sharpSpike = shortSpikeNum(data,minNumSamples)
    sampEN = sample_entropy(data[0])
    multiscaleEN = multiscale_entropy(data[0])
    permEN = permutation_entropy(data[0])
    specEN = spectral_entropy(data[0],fs)
    svdEN = svd_entropy(data[0])
    appEN = app_entropy(data[0])
    lziv_comp = lziv(data[0])
    petrosian_fd = petrosian(data[0])
    katz_fd = katz(data[0])
    dfa_fd = dfa(data[0])

    # Frequency Domain Features
    print("Calculating frequency domain features...")
    shannonRes_delta = shannonEntropy(delta_data, bin_min=-200, bin_max=200, binWidth=2)
    shannonRes_theta = shannonEntropy(theta_data, bin_min=-200, bin_max=200, binWidth=2)
    shannonRes_alpha = shannonEntropy(alpha_data, bin_min=-200, bin_max=200, binWidth=2)
    shannonRes_beta = shannonEntropy(beta_data, bin_min=-200, bin_max=200, binWidth=2)
    shannonRes_gamma = shannonEntropy(gamma_data, bin_min=-200, bin_max=200, binWidth=2)
    LyapunovRes_delta = lyapunov(delta_data)
    LyapunovRes_theta = lyapunov(theta_data)
    LyapunovRes_alpha = lyapunov(alpha_data)
    LyapunovRes_beta = lyapunov(beta_data)
    LyapunovRes_gamma = lyapunov(gamma_data)
    HiguchiFD_Res_delta  = hFD(delta_data,3)
    HiguchiFD_Res_theta  = hFD(theta_data,3)
    HiguchiFD_Res_alpha  = hFD(alpha_data,3)
    HiguchiFD_Res_beta  = hFD(beta_data,3)
    HiguchiFD_Res_gamma  = hFD(gamma_data,3)
    HjorthMob_delta, HjorthComp_delta = hjorthParameters(delta_data)
    HjorthMob_theta, HjorthComp_theta = hjorthParameters(theta_data)
    HjorthMob_alpha, HjorthComp_alpha = hjorthParameters(alpha_data)
    HjorthMob_beta, HjorthComp_beta = hjorthParameters(beta_data)
    HjorthMob_gamma, HjorthComp_gamma = hjorthParameters(gamma_data)
    stdRes_delta = eegStd(delta_data)
    stdRes_theta = eegStd(theta_data)
    stdRes_alpha = eegStd(alpha_data)
    stdRes_beta = eegStd(beta_data)
    stdRes_gamma = eegStd(gamma_data)
    regularity_res_delta = eegRegularity(delta_data,fs)
    regularity_res_theta = eegRegularity(theta_data,fs)
    regularity_res_alpha = eegRegularity(alpha_data,fs)
    regularity_res_beta = eegRegularity(beta_data,fs)
    regularity_res_gamma = eegRegularity(gamma_data,fs)
    print('regularity_res_delta shape: ',regularity_res_delta.shape)
    print('regularity_res_theta shape: ',regularity_res_theta.shape)
    print('regularity_res_alpha shape: ',regularity_res_alpha.shape)
    print('regularity_res_beta shape: ',regularity_res_beta.shape)
    print('regularity_res_gamma shape: ',regularity_res_gamma.shape)
    volt05_res_delta = eegVoltage(delta_data,5)
    volt05_res_theta = eegVoltage(theta_data,5)
    volt05_res_alpha = eegVoltage(alpha_data,5)
    volt05_res_beta = eegVoltage(beta_data,5)
    volt05_res_gamma = eegVoltage(gamma_data,5)
    volt10_res_delta = eegVoltage(delta_data,10)
    volt10_res_theta = eegVoltage(theta_data,10)
    volt10_res_alpha = eegVoltage(alpha_data,10)
    volt10_res_beta = eegVoltage(beta_data,10)
    volt10_res_gamma = eegVoltage(gamma_data,10)
    volt20_res_delta = eegVoltage(delta_data,20)
    volt20_res_theta = eegVoltage(theta_data,20)
    volt20_res_alpha = eegVoltage(alpha_data,20)
    volt20_res_beta = eegVoltage(beta_data,20)
    volt20_res_gamma = eegVoltage(gamma_data,20)
    volt50_res_delta = eegVoltage(delta_data,50)
    volt50_res_theta = eegVoltage(theta_data,50)
    volt50_res_alpha = eegVoltage(alpha_data,50)
    volt50_res_beta = eegVoltage(beta_data,50)
    volt50_res_gamma = eegVoltage(gamma_data,50)
    volt100_res_delta = eegVoltage(delta_data,100)
    volt100_res_theta = eegVoltage(theta_data,100)
    volt100_res_alpha = eegVoltage(alpha_data,100)
    volt100_res_beta = eegVoltage(beta_data,100)
    volt100_res_gamma = eegVoltage(gamma_data,100)
    print('volt100_res_delta shape: ',volt100_res_delta.shape)
    print('volt100_res_theta shape: ',volt100_res_theta.shape)
    print('volt100_res_alpha shape: ',volt100_res_alpha.shape)
    print('volt100_res_beta shape: ',volt100_res_beta.shape)
    print('volt100_res_gamma shape: ',volt100_res_gamma.shape)
    print('volt50_res_delta shape: ',volt50_res_delta.shape)
    print('volt50_res_theta shape: ',volt50_res_theta.shape)
    print('volt50_res_alpha shape: ',volt50_res_alpha.shape)
    print('volt50_res_beta shape: ',volt50_res_beta.shape)
    print('volt50_res_gamma shape: ',volt50_res_gamma.shape)
    print('volt20_res_delta shape: ',volt20_res_delta.shape)
    print('volt20_res_theta shape: ',volt20_res_theta.shape)
    print('volt20_res_alpha shape: ',volt20_res_alpha.shape)
    print('volt20_res_beta shape: ',volt20_res_beta.shape)
    print('volt20_res_gamma shape: ',volt20_res_gamma.shape)
    print('volt10_res_delta shape: ',volt10_res_delta.shape)
    print('volt10_res_theta shape: ',volt10_res_theta.shape)
    print('volt10_res_alpha shape: ',volt10_res_alpha.shape)
    print('volt10_res_beta shape: ',volt10_res_beta.shape)
    print('volt10_res_gamma shape: ',volt10_res_gamma.shape)
    print('volt05_res_delta shape: ',volt05_res_delta.shape)
    print('volt05_res_theta shape: ',volt05_res_theta.shape)
    print('volt05_res_alpha shape: ',volt05_res_alpha.shape)
    print('volt05_res_beta shape: ',volt05_res_beta.shape)
    print('volt05_res_gamma shape: ',volt05_res_gamma.shape)
    sharpSpike_delta = shortSpikeNum(delta_data,minNumSamples)
    sharpSpike_theta = shortSpikeNum(theta_data,minNumSamples)
    sharpSpike_alpha = shortSpikeNum(alpha_data,minNumSamples)
    sharpSpike_beta = shortSpikeNum(beta_data,minNumSamples)
    sharpSpike_gamma = shortSpikeNum(gamma_data,minNumSamples)
    sampEN_delta = sample_entropy(delta_data[0])
    sampEN_theta = sample_entropy(theta_data[0])
    sampEN_alpha = sample_entropy(alpha_data[0])
    sampEN_beta = sample_entropy(beta_data[0])
    sampEN_gamma = sample_entropy(gamma_data[0])
    multiscaleEN_delta = multiscale_entropy(delta_data[0])
    multiscaleEN_theta = multiscale_entropy(theta_data[0])
    multiscaleEN_alpha = multiscale_entropy(alpha_data[0])
    multiscaleEN_beta = multiscale_entropy(beta_data[0])
    multiscaleEN_gamma = multiscale_entropy(gamma_data[0])
    permEN_delta = permutation_entropy(delta_data[0])
    permEN_theta = permutation_entropy(theta_data[0])
    permEN_alpha = permutation_entropy(alpha_data[0])
    permEN_beta = permutation_entropy(beta_data[0])
    permEN_gamma = permutation_entropy(gamma_data[0])
    specEN_delta = spectral_entropy(delta_data[0],fs)
    specEN_theta = spectral_entropy(theta_data[0],fs)
    specEN_alpha = spectral_entropy(alpha_data[0],fs)
    specEN_beta = spectral_entropy(beta_data[0],fs)
    specEN_gamma = spectral_entropy(gamma_data[0],fs)
    svdEN_delta = svd_entropy(delta_data[0])
    svdEN_theta = svd_entropy(theta_data[0])
    svdEN_alpha = svd_entropy(alpha_data[0])
    svdEN_beta = svd_entropy(beta_data[0])
    svdEN_gamma = svd_entropy(gamma_data[0])
    appEN_delta = app_entropy(delta_data[0])
    appEN_theta = app_entropy(theta_data[0])
    appEN_alpha = app_entropy(alpha_data[0])
    appEN_beta = app_entropy(beta_data[0])
    appEN_gamma = app_entropy(gamma_data[0])
    lziv_delta = lziv(delta_data[0])
    lziv_theta = lziv(theta_data[0])
    lziv_alpha = lziv(alpha_data[0])
    lziv_beta = lziv(beta_data[0])
    lziv_gamma = lziv(gamma_data[0])
    petrosianFD_delta = petrosian(delta_data[0])
    petrosianFD_theta = petrosian(theta_data[0])
    petrosianFD_alpha = petrosian(alpha_data[0])
    petrosianFD_beta = petrosian(beta_data[0])
    petrosianFD_gamma = petrosian(gamma_data[0])
    katzFD_delta = katz(delta_data[0])
    katzFD_theta = katz(theta_data[0])
    katzFD_alpha = katz(alpha_data[0])
    katzFD_beta = katz(beta_data[0])
    katzFD_gamma = katz(gamma_data[0])
    dfa_delta = dfa(delta_data[0])
    dfa_theta = dfa(theta_data[0])
    dfa_alpha = dfa(alpha_data[0])
    dfa_beta = dfa(beta_data[0])
    dfa_gamma = dfa(gamma_data[0])
    
    label = np.array(['clean' for i in range(len(shannonRes))])


    # print shape of all the features
    print('shannonRes shape: ',shannonRes.shape)
    print('lyapunovRes shape: ',LyapunovRes.shape)
    print('higuchiRes shape: ',HiguchiFD_Res.shape)
    print('hjorthmob shape: ',HjorthMob.shape)
    print('hjorthcomp shape: ',HjorthComp.shape)
    print('medianFreq shape: ',medianFreqRes.shape)
    print('delta_rbandpwr shape: ',delta_rbandpwr.shape)
    print('theta_rbandpwr shape: ',theta_rbandpwr.shape)
    print('alpha_rbandpwr shape: ',alpha_rbandpwr.shape)
    print('beta_rbandpwr shape: ',beta_rbandpwr.shape)
    print('gamma_rbandpwr shape: ',gamma_rbandpwr.shape)
    print('std shape: ',stdRes.shape)
    print('delta_theta_ratio shape: ',delta_theta_ratio.shape)
    print('theta_delta_ratio shape: ',theta_delta_ratio.shape)
    print('delta_alpha_ratio shape: ',delta_alpha_ratio.shape)
    print('alpha_delta_ratio shape: ',alpha_delta_ratio.shape)
    print('delta_beta_ratio shape: ',delta_beta_ratio.shape)
    print('beta_delta_ratio shape: ',beta_delta_ratio.shape)
    print('delta_gamma_ratio shape: ',delta_gamma_ratio.shape)
    print('gamma_delta_ratio shape: ',gamma_delta_ratio.shape)
    print('theta_alpha_ratio shape: ',theta_alpha_ratio.shape)
    print('alpha_theta_ratio shape: ',alpha_theta_ratio.shape)
    print('theta_beta_ratio shape: ',theta_beta_ratio.shape)
    print('beta_theta_ratio shape: ',beta_theta_ratio.shape)
    print('theta_gamma_ratio shape: ',theta_gamma_ratio.shape)
    print('gamma_theta_ratio shape: ',gamma_theta_ratio.shape)
    print('alpha_beta_ratio shape: ',alpha_beta_ratio.shape)
    print('beta_alpha_ratio shape: ',beta_alpha_ratio.shape)
    print('alpha_gamma_ratio shape: ',alpha_gamma_ratio.shape)
    print('gamma_alpha_ratio shape: ',gamma_alpha_ratio.shape)
    print('beta_gamma_ratio shape: ',beta_gamma_ratio.shape)
    print('gamma_beta_ratio shape: ',gamma_beta_ratio.shape)
    print('regularity shape: ',regularity_res.shape)
    print('volt05 shape: ',volt05_res.shape)
    print('volt10 shape: ',volt10_res.shape)
    print('volt20 shape: ',volt20_res.shape)
    print('volt50 shape: ',volt50_res.shape)
    print('volt100 shape: ',volt100_res.shape)
    print('sharpSpike shape: ',sharpSpike.shape)
    print('sampEN:',sampEN.shape)
    print('multiscaleEN:',multiscaleEN.shape)
    print('permEN:',permEN.shape)
    print('specEN:',specEN.shape)
    print('svdEN:',svdEN.shape)
    print('appEN:',appEN.shape)
    print('lziv:',lziv_comp.shape)
    print('petrosianFD:',petrosian_fd.shape)
    print('katzFD:',katz_fd.shape)
    print('dfa:',dfa_fd.shape)
    

    # Create a list of all the features names
    feat_names = ['shannonRes_'+data_name, 'lyapunovRes_'+data_name, 'higuchiRes_'+data_name,
                    'hjorthmob_'+data_name, 'hjorthcomp_'+data_name, 'medianFreq_'+data_name,
                    'delta_rbandpwr_'+data_name, 'theta_rbandpwr_'+data_name, 'alpha_rbandpwr_'+data_name,
                    'beta_rbandpwr_'+data_name, 'gamma_rbandpwr_'+data_name, 'std_'+data_name,
                    'delta_theta_ratio_'+data_name, 'theta_delta_ratio_'+data_name, 'delta_alpha_ratio_'+data_name,
                    'alpha_delta_ratio_'+data_name, 'delta_beta_ratio_'+data_name, 'beta_delta_ratio_'+data_name,
                    'delta_gamma_ratio_'+data_name, 'gamma_delta_ratio_'+data_name, 'theta_alpha_ratio_'+data_name,
                    'alpha_theta_ratio_'+data_name, 'theta_beta_ratio_'+data_name, 'beta_theta_ratio_'+data_name,
                    'theta_gamma_ratio_'+data_name, 'gamma_theta_ratio_'+data_name, 'alpha_beta_ratio_'+data_name,
                    'beta_alpha_ratio_'+data_name, 'alpha_gamma_ratio_'+data_name, 'gamma_alpha_ratio_'+data_name,
                    'beta_gamma_ratio_'+data_name, 'gamma_beta_ratio_'+data_name, 'regularity_'+data_name,
                    'volt05_res_'+data_name, 'volt10_res_'+data_name, 'volt20_res_'+data_name, 'volt50_res_'+data_name, 'volt100_res_'+data_name,
                    'sharpSpike_'+data_name, 'sampEN_'+data_name, 'multiscaleEN_'+data_name, 'permEN_'+data_name, 'specEN_'+data_name,
                    'svdEN_'+data_name, 'appEN_'+data_name, 'lziv_'+data_name, 'petrosianFD_'+data_name, 'katzFD_'+data_name, 'dfa_'+data_name,
                    'shannonRes_delta_'+data_name, 'shannonRes_theta_'+data_name, 'shannonRes_alpha_'+data_name, 'shannonRes_beta_'+data_name, 'shannonRes_gamma_'+data_name,
                    'lyapunovRes_delta_'+data_name, 'lyapunovRes_theta_'+data_name, 'lyapunovRes_alpha_'+data_name, 'lyapunovRes_beta_'+data_name, 'lyapunovRes_gamma_'+data_name,
                    'higuchiRes_delta_'+data_name, 'higuchiRes_theta_'+data_name, 'higuchiRes_alpha_'+data_name, 'higuchiRes_beta_'+data_name, 'higuchiRes_gamma_'+data_name,
                    'hjorthmob_delta_'+data_name, 'hjorthmob_theta_'+data_name, 'hjorthmob_alpha_'+data_name, 'hjorthmob_beta_'+data_name, 'hjorthmob_gamma_'+data_name,
                    'hjorthcomp_delta_'+data_name, 'hjorthcomp_theta_'+data_name, 'hjorthcomp_alpha_'+data_name, 'hjorthcomp_beta_'+data_name, 'hjorthcomp_gamma_'+data_name,
                    'std_delta_'+data_name, 'std_theta_'+data_name, 'std_alpha_'+data_name, 'std_beta_'+data_name, 'std_gamma_'+data_name,
                    'regularity_delta_'+data_name, 'regularity_theta_'+data_name, 'regularity_alpha_'+data_name, 'regularity_beta_'+data_name, 'regularity_gamma_'+data_name,
                    'volt05_res_delta_'+data_name, 'volt05_res_theta_'+data_name, 'volt05_res_alpha_'+data_name, 'volt05_res_beta_'+data_name, 'volt05_res_gamma_'+data_name,
                    'volt10_res_delta_'+data_name, 'volt10_res_theta_'+data_name, 'volt10_res_alpha_'+data_name, 'volt10_res_beta_'+data_name, 'volt10_res_gamma_'+data_name,
                    'volt20_res_delta_'+data_name, 'volt20_res_theta_'+data_name, 'volt20_res_alpha_'+data_name, 'volt20_res_beta_'+data_name, 'volt20_res_gamma_'+data_name,
                    'volt50_res_delta_'+data_name, 'volt50_res_theta_'+data_name, 'volt50_res_alpha_'+data_name, 'volt50_res_beta_'+data_name, 'volt50_res_gamma_'+data_name,
                    'volt100_res_delta_'+data_name, 'volt100_res_theta_'+data_name, 'volt100_res_alpha_'+data_name, 'volt100_res_beta_'+data_name, 'volt100_res_gamma_'+data_name,
                    'sharpSpike_delta_'+data_name, 'sharpSpike_theta_'+data_name, 'sharpSpike_alpha_'+data_name, 'sharpSpike_beta_'+data_name, 'sharpSpike_gamma_'+data_name,
                    'sampEN_delta_'+data_name, 'sampEN_theta_'+data_name, 'sampEN_alpha_'+data_name, 'sampEN_beta_'+data_name, 'sampEN_gamma_'+data_name,
                    'multiscaleEN_delta_'+data_name, 'multiscaleEN_theta_'+data_name, 'multiscaleEN_alpha_'+data_name, 'multiscaleEN_beta_'+data_name, 'multiscaleEN_gamma_'+data_name,
                    'permEN_delta_'+data_name, 'permEN_theta_'+data_name, 'permEN_alpha_'+data_name, 'permEN_beta_'+data_name, 'permEN_gamma_'+data_name,
                    'specEN_delta_'+data_name, 'specEN_theta_'+data_name, 'specEN_alpha_'+data_name, 'specEN_beta_'+data_name, 'specEN_gamma_'+data_name,
                    'svdEN_delta_'+data_name, 'svdEN_theta_'+data_name, 'svdEN_alpha_'+data_name, 'svdEN_beta_'+data_name, 'svdEN_gamma_'+data_name,
                    'appEN_delta_'+data_name, 'appEN_theta_'+data_name, 'appEN_alpha_'+data_name, 'appEN_beta_'+data_name, 'appEN_gamma_'+data_name,
                    'lziv_delta_'+data_name, 'lziv_theta_'+data_name, 'lziv_alpha_'+data_name, 'lziv_beta_'+data_name, 'lziv_gamma_'+data_name,
                    'petrosianFD_delta_'+data_name, 'petrosianFD_theta_'+data_name, 'petrosianFD_alpha_'+data_name, 'petrosianFD_beta_'+data_name, 'petrosianFD_gamma_'+data_name,
                    'katzFD_delta_'+data_name, 'katzFD_theta_'+data_name, 'katzFD_alpha_'+data_name, 'katzFD_beta_'+data_name, 'katzFD_gamma_'+data_name,
                    'dfa_delta_'+data_name, 'dfa_theta_'+data_name, 'dfa_alpha_'+data_name, 'dfa_beta_'+data_name, 'dfa_gamma_'+data_name,'label']

    conc_data = np.vstack((shannonRes, LyapunovRes,HiguchiFD_Res, HjorthMob, HjorthComp, medianFreqRes, delta_rbandpwr, theta_rbandpwr, alpha_rbandpwr, beta_rbandpwr, gamma_rbandpwr,
                        stdRes, delta_theta_ratio, theta_delta_ratio, delta_alpha_ratio, alpha_delta_ratio, delta_beta_ratio, beta_delta_ratio, delta_gamma_ratio, gamma_delta_ratio,
                        theta_alpha_ratio, alpha_theta_ratio, theta_beta_ratio, beta_theta_ratio, theta_gamma_ratio, gamma_theta_ratio, alpha_beta_ratio, beta_alpha_ratio, alpha_gamma_ratio, gamma_alpha_ratio,
                        beta_gamma_ratio, gamma_beta_ratio,regularity_res, volt05_res, volt10_res, volt20_res, volt50_res, volt100_res, sharpSpike, sampEN, multiscaleEN, permEN, specEN, svdEN, appEN, lziv_comp,
                        petrosian_fd, katz_fd, dfa_fd, shannonRes_delta, shannonRes_theta, shannonRes_alpha, shannonRes_beta, shannonRes_gamma, LyapunovRes_delta, LyapunovRes_theta, LyapunovRes_alpha, LyapunovRes_beta, LyapunovRes_gamma,
                        HiguchiFD_Res_delta, HiguchiFD_Res_theta, HiguchiFD_Res_alpha, HiguchiFD_Res_beta, HiguchiFD_Res_gamma, HjorthMob_delta, HjorthMob_theta, HjorthMob_alpha, HjorthMob_beta, HjorthMob_gamma,
                            HjorthComp_delta, HjorthComp_theta, HjorthComp_alpha, HjorthComp_beta, HjorthComp_gamma, stdRes_delta, stdRes_theta, stdRes_alpha, stdRes_beta, stdRes_gamma,
                                regularity_res_delta, regularity_res_theta, regularity_res_alpha, regularity_res_beta, regularity_res_gamma, volt05_res_delta, volt05_res_theta, volt05_res_alpha, volt05_res_beta, volt05_res_gamma,
                                volt10_res_delta, volt10_res_theta, volt10_res_alpha, volt10_res_beta, volt10_res_gamma, volt20_res_delta, volt20_res_theta, volt20_res_alpha, volt20_res_beta, volt20_res_gamma,
                                volt50_res_delta, volt50_res_theta, volt50_res_alpha, volt50_res_beta, volt50_res_gamma, volt100_res_delta, volt100_res_theta, volt100_res_alpha, volt100_res_beta, volt100_res_gamma,
                                sharpSpike_delta, sharpSpike_theta, sharpSpike_alpha, sharpSpike_beta, sharpSpike_gamma, sampEN_delta, sampEN_theta, sampEN_alpha, sampEN_beta, sampEN_gamma,
                                multiscaleEN_delta, multiscaleEN_theta, multiscaleEN_alpha, multiscaleEN_beta, multiscaleEN_gamma, permEN_delta, permEN_theta, permEN_alpha, permEN_beta, permEN_gamma,
                                specEN_delta, specEN_theta, specEN_alpha, specEN_beta, specEN_gamma, svdEN_delta, svdEN_theta, svdEN_alpha, svdEN_beta, svdEN_gamma, appEN_delta, appEN_theta, appEN_alpha, appEN_beta, appEN_gamma,
                                lziv_delta, lziv_theta, lziv_alpha, lziv_beta, lziv_gamma, petrosianFD_delta, petrosianFD_theta, petrosianFD_alpha, petrosianFD_beta, petrosianFD_gamma, katzFD_delta, katzFD_theta, katzFD_alpha, katzFD_beta, katzFD_gamma,
                                dfa_delta, dfa_theta, dfa_alpha, dfa_beta, dfa_gamma,label))

    conc_data = conc_data.T

    conc_data = pd.DataFrame(conc_data, columns=feat_names)
    print('Features extracted')
    return conc_data

