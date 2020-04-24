import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as Data

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

import warnings 
warnings.filterwarnings("ignore")


# In[3]:


# check CPU or GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('torch version: ', torch.__version__)
print('GPU State:', device)


# In[4]:


# Read Dataset
def read_bci_data():
    S4b_train = np.load('/kaggle/input/eegnet/S4b_train.npz')
    X11b_train = np.load('/kaggle/input/eegnet/X11b_train.npz')
    S4b_test = np.load('/kaggle/input/eegnet/S4b_test.npz')
    X11b_test = np.load('/kaggle/input/eegnet/X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    # Datatype - float32 (both X and Y)
    # X.shape - (#samples, 1, #channels, #timepoints)
    # Y.shape - (#samples)
    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label

X_train, y_train, X_test, y_test = read_bci_data()


# In[6]:


# EEGNet using Pytorch
class EEGNet(nn.Module):
    def __init__(self, AF=nn.ELU(alpha=1)):
        super(EEGNet, self).__init__()
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, 1e-05)
        )      
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, 1e-05),
            AF,
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, 1e-05),
            AF,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )
        
    def forward(self, x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classify(x.view(len(x), -1))
        return x    
EEGNet()


# In[7]:


# DeepConvNet using Pytorch
class DeepConvNet(nn.Module):
    def __init__(self, AF=nn.ELU(alpha=1)):
        super(DeepConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(25, 25, kernel_size=(2, 5), padding=(0, 0), bias=False),
            nn.BatchNorm2d(25, 1e-05),
            AF,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )    
        self.layer2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(50),
            AF,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(100),
            AF,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(200),
            AF,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=9200, out_features=2)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc(x.view(len(x), -1))
        return x
DeepConvNet()


# In[10]:


# Evaluatoin Function
def evaluate(model, X, y):
    model.eval()
    inputs = Variable(torch.Tensor(X).cuda())
    pred = model(inputs)
    
    return accuracy_score(y, np.array(pred.data.cpu().numpy()).argmax(axis=1))


# In[11]:


# Tensor Format Transformation Function
def loader(X, y, batch_size):
    dataset = Data.TensorDataset(torch.Tensor(X), torch.Tensor(y))
    torch.cuda.manual_seed_all(3023)
    return Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )


# In[12]:


# Model Traning
def model_train(net, train, test, epochs, batch_size, scheduler):
    train_acc = []
    test_acc = []

    for epoch in range(epochs+1):  
        net.train(mode=True)
        train_loss = 0.0
        for x, y in train:
            x_batch, y_batch = Variable(x.cuda()), Variable(y.cuda()).long()
            
            optimizer.zero_grad() # zero the parameter gradients
            outputs = net(x_batch) # forward
            loss = criterion(outputs, y_batch) # calculate loss
            loss.backward() # backward
            optimizer.step()
            train_loss += loss.data

        train_acc.append(evaluate(net, X_train, y_train))
        test_acc.append(evaluate(net, X_test, y_test))
        scheduler.step()            

        # accuracy result every 50 epoch
        if epoch % 50 == 0:
            print("\nEpoch ", epoch)
            print("(Training Loss) -", train_loss.data.cpu().numpy())
            print("(Train) - ", evaluate(net, X_train, y_train))
            print("(Test) - ", evaluate(net, X_test, y_test))
            
    return train_acc, test_acc


# In[25]:


# EEGNet Start
print('=== EEGNet start ============================================================')
plt.figure(figsize=(12, 6))
epochs = 500
batch_size = 256
lr = 0.002
step_size = 150
gamma = 0.9 # gamma of scheduler 
train = loader(X_train, y_train, batch_size)
test = loader(X_test, y_test, batch_size)

activation_box = ['ReLU', 'LeakyReLU', 'ELU', 'PReLU'] # activation function you want to try
train_acc, test_acc = {}, {}
best_train_acc, best_test_acc = {}, {}

for af in activation_box:
    print('\n[Activation function] : %s ------------------------------'%af)
    AF = eval('nn.' + af + '()')
    net = EEGNet(AF=AF).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    train_acc[af+'_train'], test_acc[af+'_test'] = model_train(net, train, test, epochs=epochs, batch_size=batch_size, scheduler=scheduler)
    plt.plot(train_acc[af+'_train'], label=af+'_train')
    plt.plot(test_acc[af+'_test'], label=af+'_test')
    plt.legend()


# In[26]:


# DeepConvNet Start
plt.figure(figsize=(12, 6))
epochs = 500
batch_size = 256
lr = 0.002
step_size = 150
gamma = 0.9 # gamma of scheduler 
train = loader(X_train, y_train, batch_size)
test = loader(X_test, y_test, batch_size)

activation_box = ['ReLU', 'LeakyReLU', 'ELU', 'PReLU']# activation function you want to try
train_acc, test_acc = {}, {}
best_train_acc, best_test_acc = {}, {}

for af in activation_box:
    AF = eval('nn.' + af + '()')
    net = DeepConvNet(AF=AF).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    train_acc[af+'_train'], test_acc[af+'_test'] = model_train(net, train, test, epochs=epochs, batch_size=batch_size, scheduler=scheduler)
    plt.plot(train_acc[af+'_train'], label=af+'_train')
    plt.plot(test_acc[af+'_test'], label=af+'_test')
    plt.legend()





