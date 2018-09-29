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
# feed traindata.wav into wavfile

wavfile = ".wav"
x, sr = librosa.load(wavfile, sr = 16000, mono = 'True')
C = np.abs(librosa.cqt(x[:int(2.68*sr)], sr=sr, n_bins=84))
