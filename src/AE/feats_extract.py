# importing libraries
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.style as ms
#ms.use('seaborn-muted')
#%matplotlib inline
#import IPython.display
import librosa
#import librosa.display
#https://musicinformationretrieval.com/ipython_audio.html
import os,glob,h5py
from multiprocessing import Pool
from tqdm import tqdm

# implement iterator to load wav files into x
cqarray = np.zeros((1,84,84))
count=0
def CQTgenerate(filenames_):
    x,sr = librosa.load(filenames_, sr = 16000, mono = 'True')
    c = np.abs(librosa.cqt(x[:int(2.68*sr)], sr=sr, n_bins=84))
    c = np.reshape(c,(1,84,84))
    return c

if __name__ == '__main__':
    os.chdir("D:\\ImplementAI_data\\nsynth-train")
    filenames = glob.glob("audio\\vocal*.wav")
    print (len(filenames))
    agents = 4
    chunksize = 20
    with Pool(processes=agents) as pool:
        cqarray = list(tqdm(pool.imap(CQTgenerate,filenames),total=len(filenames)))
    cqarray = np.array(cqarray)
    print(np.shape(cqarray),cqarray.dtype)
    h5f = h5py.File('train_data_vocal.h5','w')
    h5f.create_dataset('train',data = cqarray)