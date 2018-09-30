import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from logger import Logger
import h5py
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

'''
# MNIST Dataset 
dataset = dsets.MNIST(root='./data', 
                      train=True, 
                      transform=transforms.ToTensor(),  
                      download=True)

# Data Loader (Input Pipeline)
data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                          batch_size=100, 
                                          shuffle=True)
'''

class SpectrogramData(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, labels, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataTensor = data
        self.labelTensor = labels
        self.transform = transform

    def __len__(self):
        return len(self.labelTensor)

    def __getitem__(self, idx):
        #img_name = os.path.join(self.root_dir,
         #                       self.landmarks_frame.iloc[idx, 0])
        #image = io.imread(img_name)
        #landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        if self.transform:
#            print(np.shape(self.dataTensor[idx]))
            sample = np.reshape(self.dataTensor[idx],(1,84,1))
            sample = self.transform(sample)
            
            
        sample = sample.type(torch.FloatTensor), self.labelTensor[idx].astype(int)
        return sample


'''
# read hdf5 files and create a compact hdf5 file with all classes, some samples
file_count = 0
filenames = glob.glob('D:\\ImplementAI_data\\nsynth-train\\*.h5')
print(filenames)
h5pySizes = [12675,8773,32690,51821,34201,34477,13911,19474,5501,10208] #65474,
for file in glob.glob('D:\\ImplementAI_data\\nsynth-train\\*.h5'):    
    file_count=file_count+1
    selection_index = np.random.randint(0,h5pySizes[file_count-1],int(0.25*h5pySizes[file_count-1]))
    data_f = h5py.File(file,'r');
#    data = np.mean(np.array(data_f['train']),axis=3)
    data = np.array(data_f['train'])
    labels = file_count*np.ones((np.shape(data)[0],))
    data = data[selection_index]
    labels = labels[selection_index]
    print(np.shape(data),np.shape(labels))
    if file_count==1:
        Data_tensor = data
        Label_tensor = labels
    else:
        Data_tensor = np.append(Data_tensor,data,axis=0)
        Label_tensor = np.append(Label_tensor,labels,axis=0)
    print(np.shape(Data_tensor),np.shape(Label_tensor))
h5f = h5py.File('D:\\ImplementAI_data\\nsynth-train\\train_data_allClasses.h5','w')
h5f.create_dataset('Data',data = Data_tensor)
h5f.create_dataset('labels',data = Label_tensor)
'''
data_f = h5py.File('D:\\ImplementAI_data\\train_data_allClasses_Time_mean.h5','r');
data = np.array(data_f['Data'])
data = data/np.max(data)
labels = np.array(data_f['labels'])
print(type(data),type(labels),data.dtype,labels.dtype)

dataset = SpectrogramData(data,labels,transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                          batch_size=1, 
                                          shuffle=False)

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)    

def accuracy_metric(output,targets):
    _, predicted = torch.max(output.data,1)
#    print(output.size(),targets.size(),predicted.size())
    accuracy = (predicted==targets.data.type(torch.cuda.LongTensor)).sum()/len(targets)
    return accuracy

#Encoder
pretrained_weights = torch.load('Q_encoder_weights_2.pt')
class Q_net(nn.Module):  
    def __init__(self,X_dim,N,z_dim):
        super(Q_net, self).__init__()
        self.xdim = X_dim
        self.lin1 = nn.Linear(X_dim, N)
#        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, int(N/2))
        self.lin3gauss = nn.Linear(int(N/2), z_dim)
    def forward(self, x):
        x = x.view(-1,self.xdim)
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
#        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
#        x = F.relu(x)
        x = F.dropout(self.lin3(x), p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        return xgauss
    
class Q_convNet(nn.Module):  
    def __init__(self,X_dim,N,z_dim):
        super(Q_net, self).__init__()
        self.xdim = X_dim
        self.conv1 = nn.Conv1d(1,16,7,stride=3,padding=3) # 16X28
        self.conv2 = nn.Conv1d(16,32,5,stride=2,padding=2) # 32X14
        self.lin1 = nn.Linear(32*14, N)
#        self.lin2 = nn.Linear(N, N)
#        self.lin3 = nn.Linear(N, int(N/2))
        self.lin3gauss = nn.Linear(N, z_dim)
    def forward(self, x):
        x = x.view(-1,1,self.xdim)
        x = F.dropout(self.conv1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.conv2(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        return xgauss

# Decoder
class P_net(nn.Module):  
    def __init__(self,X_dim,N,z_dim):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, int(N/2))
        self.lin2 = nn.Linear(int(N/2), N)
#        self.lin3 = nn.Linear(N, N)
        self.lin4 = nn.Linear(N, X_dim)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
#        x = F.dropout(self.lin3(x), p=0.25, training=self.training)
        x = self.lin4(x)
        return F.sigmoid(x)

class P_convNet(nn.Module):  
    def __init__(self,X_dim,N,z_dim):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, 32*14)
#        self.lin3 = nn.Linear(N, N)
        self.lin4 = nn.Linear(N, X_dim)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
#        x = F.dropout(self.lin3(x), p=0.25, training=self.training)
        x = self.lin4(x)
        return F.sigmoid(x)

# Discriminator
class D_net_gauss(nn.Module):  
    def __init__(self,N,z_dim):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
#        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, int(N/2))
        self.lin4 = nn.Linear(int(N/2), 10)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.5, training=self.training)
        x = F.relu(x)
#        x = F.dropout(self.lin2(x), p=0.5, training=self.training)
#        x = F.relu(x)
        x = F.dropout(self.lin3(x), p=0.5, training=self.training)
        x = F.relu(x)
#        return F.sigmoid(self.lin3(x))        
        return F.log_softmax(self.lin4(x))

EPS = 1e-15
z_red_dims = 64
Q = Q_net(84,128,z_red_dims).cuda()
P = P_net(84,128,z_red_dims).cuda()
D_gauss = D_net_gauss(32,z_red_dims).cuda()
Q.load_state_dict(pretrained_weights)
# Set the logger
logger = Logger('./logs/z_120_fixed_LR_2')

# Set learning rates
gen_lr = 0.001
reg_lr = 0.001

#encode/decode optimizers
optim_P = torch.optim.Adam(P.parameters(), lr=gen_lr)
optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=gen_lr)
#regularizing optimizers
optim_Q_gen = torch.optim.Adam(Q.parameters(), lr=reg_lr)
optim_D = torch.optim.Adam(D_gauss.parameters(), lr=reg_lr)
    
data_iter = iter(data_loader)
iter_per_epoch = len(data_loader)
print("iter per epoch =",iter_per_epoch)
total_step = np.shape(data)[0]
latent_vector = np.zeros((total_step,z_red_dims))

# Start training
for step in tqdm(range(total_step)):

    # Reset the data_iter
    if (step+1) % iter_per_epoch == 0:
        data_iter = iter(data_loader)
        print(step)

    # Fetch the images and labels and convert them to variables
    images, labels = next(data_iter)
#    print(images.size(),labels.size(),images.type(),labels.type())
    images, labels = to_var(images.view(images.size(0), -1)), to_var(labels)

    Q.eval()
#    z_real_gauss = Variable(torch.randn(images.size()[0], z_red_dims) * 5.).cuda()
#    D_real_gauss = D_gauss(z_real_gauss)

    z_out = Q(images)

    latent_vector[step] = to_np(z_out)

#save the Encoder
#torch.save(Q.state_dict(),'Q_encoder_weights_2.pt')
#save the latent vector
h5f = h5py.File('train_data_latent_vector.h5','w')
h5f.create_dataset('z_vec',data = latent_vector)