import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import features_helper as feat


path = '/Users/joshuaighalo/Downloads/'
filenameCEEG, filenameEMG, filenameEOG = 'EEG_all_epochs.npy', 'EMG_all_epochs.npy', 'EOG_all_epochs.npy'
dataCEEG, dataEMG, dataEOG = np.load(path + filenameCEEG), np.load(path + filenameEMG), np.load(path + filenameEOG)
fsCEEG, fsEMG, fsEOG = 256, 512, 256


dataCEEG = dataCEEG.reshape(1,dataCEEG.shape[1],dataCEEG.shape[0])
dataEMG = dataEMG.reshape(1,dataEMG.shape[1],dataEMG.shape[0])
dataEOG = dataEOG.reshape(1,dataEOG.shape[1],dataEOG.shape[0])



featCEEG = feat.compile_features(dataCEEG,fs=fsCEEG,delta=[0.5,4],theta=[4,8],
                             alpha=[8,12],beta=[12,30],gamma=[30,50],
                             data_name='CEEG', label=1)
featEMG = feat.compile_features(dataEMG,fs=fsEMG,delta=[0.5,4],theta=[4,8],
                                alpha=[8,12],beta=[12,30],gamma=[30,50],
                                data_name='EMG', label=2)
featEOG = feat.compile_features(dataEOG,fs=fsEOG,delta=[0.5,4],theta=[4,8],
                                alpha=[8,12],beta=[12,30],gamma=[30,50],
                                data_name='EOG', label=3)


# stack featCEEG, featEMG, featEOG into one dataframe
 
# stack featCEEG, featEMG, featEOG vertically
df = pd.concat([featCEEG,featEMG,featEOG],axis=0,ignore_index=True)

