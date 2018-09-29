# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
%matplotlib inline
import IPython.display
import librosa
import librosa.display
#https://musicinformationretrieval.com/ipython_audio.html


# implement iterator to load wav files into x

import os,glob,h5py
cqarray = np.zeros((1,84,84))
os.chdir(os.getcwd())
filenames = glob.glob("*.wav")
# print (filenames)
for i in range(len(filenames)):
    x,sr = librosa.load(filenames[i], sr = 16000, mono = 'True')
    c = np.abs(librosa.cqt(x[:int(2.68*sr)], sr=sr, n_bins=84))
    c = np.reshape(c,(1,84,84))
    cqarray = np.append(cqarray,c,0)

h5f = h5py.File('train_data.h5','w')
h5f.create_dataset('train',data = cqarray)
