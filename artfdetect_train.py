import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/Users/joshuaighalo/Documents/codespace/denoiser/EEGExtract')
import EEGExtract as feat

path = '/Users/joshuaighalo/Downloads/'
filenameCEEG, filenameEMG, filenameEOG = 'EEG_all_epochs.npy', 'EMG_all_epochs.npy', 'EOG_all_epochs.npy'
dataCEEG, dataEMG, dataEOG = np.load(path + filenameCEEG), np.load(path + filenameEMG), np.load(path + filenameEOG)
delta,theta,alpha,beta,gamma = [0.5,4],[4,8],[8,12],[12,30],[30,100]
fsCEEG, fsEMG, fsEOG = 256, 512, 256


dataCEEG = dataCEEG.reshape(1,dataCEEG.shape[1],dataCEEG.shape[0])
dataEMG = dataEMG.reshape(1,dataEMG.shape[1],dataEMG.shape[0])
dataEOG = dataEOG.reshape(1,dataEOG.shape[1],dataEOG.shape[0])

"""
# Subband Information Quantity
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

# Shannon Entropy
shannonResCEEG = feat.shannonEntropy(dataCEEG, bin_min=-200, bin_max=200, binWidth=2)
shannonResEMG = feat.shannonEntropy(dataEMG, bin_min=-200, bin_max=200, binWidth=2)
shannonResEOG = feat.shannonEntropy(dataEOG, bin_min=-200, bin_max=200, binWidth=2)

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



sampEN_CEEG = feat.sample_entropy(dataCEEG[0])
sampEN_EMG = feat.sample_entropy(dataEMG[0])
sampEN_EOG = feat.sample_entropy(dataEOG[0])
multiscaleEN_CEEG = feat.multiscale_entropy(dataCEEG[0])
multiscaleEN_EMG = feat.multiscale_entropy(dataEMG[0])
multiscaleEN_EOG = feat.multiscale_entropy(dataEOG[0])
permEN_CEEG = feat.permutation_entropy(dataCEEG[0])
permEN_EMG = feat.permutation_entropy(dataEMG[0])
permEN_EOG = feat.permutation_entropy(dataEOG[0])
specEN_CEEG = feat.spectral_entropy(dataCEEG[0],fsCEEG)
specEN_EMG = feat.spectral_entropy(dataEMG[0],fsEMG)
specEN_EOG = feat.spectral_entropy(dataEOG[0],fsEOG)
svdEN_CEEG = feat.svd_entropy(dataCEEG[0])
svdEN_EMG = feat.svd_entropy(dataEMG[0])
svdEN_EOG = feat.svd_entropy(dataEOG[0])
appEN_CEEG = feat.app_entropy(dataCEEG[0])
appEN_EMG = feat.app_entropy(dataEMG[0])
appEN_EOG = feat.app_entropy(dataEOG[0])
lziv_CEEG = feat.lziv(dataCEEG[0])
lziv_EMG = feat.lziv(dataEMG[0])
lziv_EOG = feat.lziv(dataEOG[0])
petrosian_CEEG = feat.petrosian(dataCEEG[0])
petrosian_EMG = feat.petrosian(dataEMG[0])
petrosian_EOG = feat.petrosian(dataEOG[0])
katz_CEEG = feat.katz(dataCEEG[0])
katz_EMG = feat.katz(dataEMG[0])
katz_EOG = feat.katz(dataEOG[0])
dfa_CEEG = feat.dfa(dataCEEG[0])
dfa_EMG = feat.dfa(dataEMG[0])
dfa_EOG = feat.dfa(dataEOG[0])


## Spectral Features

# Shannon Entropy
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
lyapunovResDeltaCEEG = feat.lyapunov(deltaCEEG)
lyapunovResThetaCEEG = feat.lyapunov(thetaCEEG)
lyapunovResAlphaCEEG = feat.lyapunov(alphaCEEG)
lyapunovResBetaCEEG = feat.lyapunov(betaCEEG)
lyapunovResGammaCEEG = feat.lyapunov(gammaCEEG)

lyapunovResDeltaEMG = feat.lyapunov(deltaEMG)
lyapunovResThetaEMG = feat.lyapunov(thetaEMG)
lyapunovResAlphaEMG = feat.lyapunov(alphaEMG)
lyapunovResBetaEMG = feat.lyapunov(betaEMG)
lyapunovResGammaEMG = feat.lyapunov(gammaEMG)

lyapunovResDeltaEOG = feat.lyapunov(deltaEOG)
lyapunovResThetaEOG = feat.lyapunov(thetaEOG)
lyapunovResAlphaEOG = feat.lyapunov(alphaEOG)
lyapunovResBetaEOG = feat.lyapunov(betaEOG)
lyapunovResGammaEOG = feat.lyapunov(gammaEOG)

# Fractal Embedding Dimension
HiguchiFD_Res_DeltaCEEG = feat.hFD(deltaCEEG,3)
HiguchiFD_Res_ThetaCEEG = feat.hFD(thetaCEEG,3)
HiguchiFD_Res_AlphaCEEG = feat.hFD(alphaCEEG,3)
HiguchiFD_Res_BetaCEEG = feat.hFD(betaCEEG,3)
HiguchiFD_Res_GammaCEEG = feat.hFD(gammaCEEG,3)

HiguchiFD_Res_DeltaEMG = feat.hFD(deltaEMG,3)
HiguchiFD_Res_ThetaEMG = feat.hFD(thetaEMG,3)
HiguchiFD_Res_AlphaEMG = feat.hFD(alphaEMG,3)
HiguchiFD_Res_BetaEMG = feat.hFD(betaEMG,3)
HiguchiFD_Res_GammaEMG = feat.hFD(gammaEMG,3)

HiguchiFD_Res_DeltaEOG = feat.hFD(deltaEOG,3)
HiguchiFD_Res_ThetaEOG = feat.hFD(thetaEOG,3)
HiguchiFD_Res_AlphaEOG = feat.hFD(alphaEOG,3)
HiguchiFD_Res_BetaEOG = feat.hFD(betaEOG,3)
HiguchiFD_Res_GammaEOG = feat.hFD(gammaEOG,3)

# Hjorth Mobility & Hjorth Complexity
HjorthMobDeltaCEEG,HjorthCompDeltaCEEG = feat.hjorthParameters(deltaCEEG)
HjorthMobThetaCEEG,HjorthCompThetaCEEG = feat.hjorthParameters(thetaCEEG)
HjorthMobAlphaCEEG,HjorthCompAlphaCEEG = feat.hjorthParameters(alphaCEEG)
HjorthMobBetaCEEG,HjorthCompBetaCEEG = feat.hjorthParameters(betaCEEG)
HjorthMobGammaCEEG,HjorthCompGammaCEEG = feat.hjorthParameters(gammaCEEG)

HjorthMobDeltaEMG,HjorthCompDeltaEMG = feat.hjorthParameters(deltaEMG)
HjorthMobThetaEMG,HjorthCompThetaEMG = feat.hjorthParameters(thetaEMG)
HjorthMobAlphaEMG,HjorthCompAlphaEMG = feat.hjorthParameters(alphaEMG)
HjorthMobBetaEMG,HjorthCompBetaEMG = feat.hjorthParameters(betaEMG)
"""







test = feat.compile_features(dataCEEG,fs=256,delta=[0.5,4],theta=[4,8],
                             alpha=[8,12],beta=[12,30],gamma=[30,50],
                             data_name='CEEG')

