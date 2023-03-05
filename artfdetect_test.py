import sys
from features_helper import *
sys.path.insert(1, '/Users/joshuaighalo/Documents/codespace/eegDementia')
from eeg_helper import *
import params as cfg

localPath = '/Users/joshuaighalo/Library/CloudStorage/OneDrive-SimonFraserUniversity(1sfu)/workspace/datasets/raw-eeg/laurel_place/cleaned_dataset'
filename = '0406_1_15042019_1047'
#filename = '0011_2_22022019_1405'

"Bruyere Dataset"
#version = 1.1
#filename = '3-24-16-29-14-cn_2_04022019_1526'
#localPath = '/Users/joshuaighalo/Downloads/brainNet_datasets/bruyere'

device = importFile.neurocatch()
version = 1.0
fileObjects = device.init(version,filename,localPath,dispIMG=False)
rawEEG = fileObjects[0]
rawEOG = fileObjects[1]
rawEEGEOG = fileObjects[2]
time = fileObjects[3]

filtering = filters()
adaptiveFilterOutput = filtering.adaptive(rawEEG,rawEOG)

plt.plot(time,rawEEG)
plt.show()