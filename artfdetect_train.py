import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/Users/joshuaighalo/Documents/codespace/denoiser/EEGExtract')
import EEGExtract as feat

path = '/Users/joshuaighalo/Downloads/'

filenameCEEG, filenameEMG, filenameEOG = 'EEG_all_epochs.npy', 'EMG_all_epochs.npy', 'EOG_all_epochs.npy'
dataCEEG, dataEMG, dataEOG = np.load(path + filenameCEEG), np.load(path + filenameEMG), np.load(path + filenameEOG)
dataCEEG = dataCEEG.reshape(1,dataCEEG.shape[1],dataCEEG.shape[0])
dataEMG = dataEMG.reshape(1,dataEMG.shape[1],dataEMG.shape[0])
dataEOG = dataEOG.reshape(1,dataEOG.shape[1],dataEOG.shape[0])

## Complexity Features

# Shannon Entropy
shannonResCEEG = feat.shannonEntropy(dataCEEG, bin_min=-200, bin_max=200, binWidth=2)
shannonResEMG = feat.shannonEntropy(dataEMG, bin_min=-200, bin_max=200, binWidth=2)
shannonResEOG = feat.shannonEntropy(dataEOG, bin_min=-200, bin_max=200, binWidth=2)

# Subband Information Quantity
delta,theta,alpha,beta,gamma = [0.5,4],[4,8],[8,12],[12,30],[30,100]
fsCEEG, fsEMG, fsEOG = 256, 512, 256
deltaCEEG = feat.filt_data(dataCEEG, delta[0], delta[1], fsCEEG)
thetaCEEG = feat.filt_data(dataCEEG, theta[0], theta[1], fsCEEG)
alphaCEEG = feat.filt_data(dataCEEG, alpha[0], alpha[1], fsCEEG)
betaCEEG = feat.filt_data(dataCEEG, beta[0], beta[1], fsCEEG)
gammaCEEG = feat.filt_data(dataCEEG, gamma[0], gamma[1], fsCEEG)
deltaEMG = feat.filt_data(dataEMG, delta[0], delta[1], fsEMG)
thetaEMG = feat.filt_data(dataEMG, theta[0], theta[1], fsEMG)
alphaEMG = feat.filt_data(dataEMG, alpha[0], alpha[1], fsEMG)
betaEMG = feat.filt_data(dataEMG, beta[0], beta[1], fsEMG)
gammaEMG = feat.filt_data(dataEMG, gamma[0], gamma[1], fsEMG)
deltaEOG = feat.filt_data(dataEOG, delta[0], delta[1], fsEOG)
thetaEOG = feat.filt_data(dataEOG, theta[0], theta[1], fsEOG)
alphaEOG = feat.filt_data(dataEOG, alpha[0], alpha[1], fsEOG)
betaEOG = feat.filt_data(dataEOG, beta[0], beta[1], fsEOG)
gammaEOG = feat.filt_data(dataEOG, gamma[0], gamma[1], fsEOG)
shannonResDeltaCEEG = feat.shannonEntropy(deltaCEEG, bin_min=-200, bin_max=200, binWidth=2)
shannonResThetaCEEG = feat.shannonEntropy(thetaCEEG, bin_min=-200, bin_max=200, binWidth=2)
shannonResAlphaCEEG = feat.shannonEntropy(alphaCEEG, bin_min=-200, bin_max=200, binWidth=2)
shannonResBetaCEEG = feat.shannonEntropy(betaCEEG, bin_min=-200, bin_max=200, binWidth=2)
shannonResGammaCEEG = feat.shannonEntropy(gammaCEEG, bin_min=-200, bin_max=200, binWidth=2)
shannonResDeltaEMG = feat.shannonEntropy(deltaEMG, bin_min=-200, bin_max=200, binWidth=2)
shannonResThetaEMG = feat.shannonEntropy(thetaEMG, bin_min=-200, bin_max=200, binWidth=2)
shannonResAlphaEMG = feat.shannonEntropy(alphaEMG, bin_min=-200, bin_max=200, binWidth=2)
shannonResBetaEMG = feat.shannonEntropy(betaEMG, bin_min=-200, bin_max=200, binWidth=2)
shannonResGammaEMG = feat.shannonEntropy(gammaEMG, bin_min=-200, bin_max=200, binWidth=2)
shannonResDeltaEOG = feat.shannonEntropy(deltaEOG, bin_min=-200, bin_max=200, binWidth=2)
shannonResThetaEOG = feat.shannonEntropy(thetaEOG, bin_min=-200, bin_max=200, binWidth=2)
shannonResAlphaEOG = feat.shannonEntropy(alphaEOG, bin_min=-200, bin_max=200, binWidth=2)
shannonResBetaEOG = feat.shannonEntropy(betaEOG, bin_min=-200, bin_max=200, binWidth=2)
shannonResGammaEOG = feat.shannonEntropy(gammaEOG, bin_min=-200, bin_max=200, binWidth=2)

# Lyapunov Exponent
LyapunovResCEEG = feat.lyapunov(dataCEEG)
LyapunovResEMG = feat.lyapunov(dataEMG)
LyapunovResEOG = feat.lyapunov(dataEOG)

# Fractal Embedding Dimension
HiguchiFD_Res_CEEG  = feat.hFD(dataCEEG,3)
HiguchiFD_Res_EMG  = feat.hFD(dataEMG,3)
HiguchiFD_Res_EOG  = feat.hFD(dataEOG,3)

# Hjorth Mobility & Hjorth Complexity
HjorthMobCEEG, HjorthCompCEEG = feat.hjorthParameters(dataCEEG)
HjorthMobEMG, HjorthCompEMG = feat.hjorthParameters(dataEMG)
HjorthMobEOG, HjorthCompEOG = feat.hjorthParameters(dataEOG)

## Category Features

# Median Frequency
medianFreqResCEEG = feat.medianFreq(dataCEEG, fsCEEG)
medianFreqResEMG = feat.medianFreq(dataEMG, fsEMG)
medianFreqResEOG = feat.medianFreq(dataEOG, fsEOG)

# δ band Power
bandPwr_delta_CEEG = feat.bandPower(dataCEEG, delta[0], delta[1], fsCEEG)
bandPwr_delta_EMG = feat.bandPower(dataEMG, delta[0], delta[1], fsEMG)
bandPwr_delta_EOG = feat.bandPower(dataEOG, delta[0], delta[1], fsEOG)

# θ band Power
bandPwr_theta_CEEG = feat.bandPower(dataCEEG, theta[0], theta[1], fsCEEG)
bandPwr_theta_EMG = feat.bandPower(dataEMG, theta[0], theta[1], fsEMG)
bandPwr_theta_EOG = feat.bandPower(dataEOG, theta[0], theta[1], fsEOG)

# α band Power
bandPwr_alpha_CEEG = feat.bandPower(dataCEEG, alpha[0], alpha[1], fsCEEG)
bandPwr_alpha_EMG = feat.bandPower(dataEMG, alpha[0], alpha[1], fsEMG)
bandPwr_alpha_EOG = feat.bandPower(dataEOG, alpha[0], alpha[1], fsEOG)

# β band Power
bandPwr_beta_CEEG = feat.bandPower(dataCEEG, beta[0], beta[1], fsCEEG)
bandPwr_beta_EMG = feat.bandPower(dataEMG, beta[0], beta[1], fsEMG)
bandPwr_beta_EOG = feat.bandPower(dataEOG, beta[0], beta[1], fsEOG)

# γ band Power
bandPwr_gamma_CEEG = feat.bandPower(dataCEEG, gamma[0], gamma[1], fsCEEG)
bandPwr_gamma_EMG = feat.bandPower(dataEMG, gamma[0], gamma[1], fsEMG)
bandPwr_gamma_EOG = feat.bandPower(dataEOG, gamma[0], gamma[1], fsEOG)

# Total Power
totalPwr_CEEG = feat.bandPower(dataCEEG, delta[0], gamma[1], fsCEEG)
totalPwr_EMG = feat.bandPower(dataEMG, delta[0], gamma[1], fsEMG)
totalPwr_EOG = feat.bandPower(dataEOG, delta[0], gamma[1], fsEOG)

# Standard Deviation
std_res_CEEG = feat.eegStd(dataCEEG)
std_res_EMG = feat.eegStd(dataEMG)
std_res_EOG = feat.eegStd(dataEOG)

# α/δ Ratio
ratio_res_CEEG = feat.eegRatio(dataCEEG,fsCEEG)
ratio_res_EMG = feat.eegRatio(dataEMG,fsEMG)
ratio_res_EOG = feat.eegRatio(dataEOG,fsEOG)

# Regularity (burst-suppression)
regularity_res_CEEG = feat.eegRegularity(dataCEEG,fsCEEG)
regularity_res_EMG = feat.eegRegularity(dataEMG,fsEMG)
regularity_res_EOG = feat.eegRegularity(dataEOG,fsEOG)

# Voltage < 5μ
volt05_res_CEEG = feat.eegVoltage(dataCEEG,voltage=5)
volt05_res_EMG = feat.eegVoltage(dataEMG,voltage=5)
volt05_res_EOG = feat.eegVoltage(dataEOG,voltage=5)

# Voltage < 10μ
volt10_res_CEEG = feat.eegVoltage(dataCEEG,voltage=10)
volt10_res_EMG = feat.eegVoltage(dataEMG,voltage=10)
volt10_res_EOG = feat.eegVoltage(dataEOG,voltage=10)

# Voltage < 20μ
volt20_res_CEEG = feat.eegVoltage(dataCEEG,voltage=20)
volt20_res_EMG = feat.eegVoltage(dataEMG,voltage=20)
volt20_res_EOG = feat.eegVoltage(dataEOG,voltage=20)

# Sharp spike
minNumSamples_CEEG = int(70*fsCEEG/1000)
minNumSamples_EMG = int(70*fsEMG/1000)
minNumSamples_EOG = int(70*fsEOG/1000)
sharpSpike_res_CEEG = feat.shortSpikeNum(dataCEEG,minNumSamples_CEEG)
sharpSpike_res_EMG = feat.shortSpikeNum(dataEMG,minNumSamples_EMG)
sharpSpike_res_EOG = feat.shortSpikeNum(dataEOG,minNumSamples_EOG)

## Connectivity Features
import numpy as np
from pyentrp import entropy as ent
import antropy as ant

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
    return output

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

def hjorth(x):
    def params_hjorth(y):
        return ant.hjorth_params(y)
    
    output = []
    for i in range(x.shape[1]):
        output.append(params_hjorth(x[:,i]))
    output = np.array(output).T
    return output

def zcr(x):
    def params_zcr(y):
        return ant.num_zerocross(x)
    
    output = []
    for i in range(x.shape[1]):
        output.append(params_zcr(x[:,i]))
    output = np.array(output).T
    output = np.mean(output,axis=1)
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



sampEN_CEEG = sample_entropy(dataCEEG[0])
sampEN_EMG = sample_entropy(dataEMG[0])
sampEN_EOG = sample_entropy(dataEOG[0])
multiscaleEN_CEEG = multiscale_entropy(dataCEEG[0])
multiscaleEN_EMG = multiscale_entropy(dataEMG[0])
multiscaleEN_EOG = multiscale_entropy(dataEOG[0])
permEN_CEEG = permutation_entropy(dataCEEG[0])
permEN_EMG = permutation_entropy(dataEMG[0])
permEN_EOG = permutation_entropy(dataEOG[0])
specEN_CEEG = spectral_entropy(dataCEEG[0],fsCEEG)
specEN_EMG = spectral_entropy(dataEMG[0],fsEMG)
specEN_EOG = spectral_entropy(dataEOG[0],fsEOG)
svdEN_CEEG = svd_entropy(dataCEEG[0])
svdEN_EMG = svd_entropy(dataEMG[0])
svdEN_EOG = svd_entropy(dataEOG[0])
appEN_CEEG = app_entropy(dataCEEG[0])
appEN_EMG = app_entropy(dataEMG[0])
appEN_EOG = app_entropy(dataEOG[0])
hjorth_CEEG = hjorth(dataCEEG[0])
hjorth_EMG = hjorth(dataEMG[0])
hjorth_EOG = hjorth(dataEOG[0])
zcr_CEEG = zcr(dataCEEG[0])
zcr_EMG = zcr(dataEMG[0])
zcr_EOG = zcr(dataEOG[0])
lziv_CEEG = lziv(dataCEEG[0])
lziv_EMG = lziv(dataEMG[0])
lziv_EOG = lziv(dataEOG[0])
petrosian_CEEG = petrosian(dataCEEG[0])
petrosian_EMG = petrosian(dataEMG[0])
petrosian_EOG = petrosian(dataEOG[0])
katz_CEEG = katz(dataCEEG[0])
katz_EMG = katz(dataEMG[0])
katz_EOG = katz(dataEOG[0])
dfa_CEEG = dfa(dataCEEG[0])
dfa_EMG = dfa(dataEMG[0])
dfa_EOG = dfa(dataEOG[0])




