import torch.nn as nn
import numpy as np
import torch

class Net(nn.Module):
    def __init__(self, input_size,hidden_size,activation='relu'):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        if activation == 'relu':
            self.relu = nn.ReLU()
        elif activation == 'tanh':
            self.relu = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, 20)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

class Dataset(torch.utils.data.Dataset):
    '''
Only for internal use by NNEstimator
    '''
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]


class NNEstimator:
    def __init__(self,lr=0.01,epochs=100,hidden_size=100,activation='relu',weight_decay=0.0,optimizer='adam',momentum=0.9):
        self.lr = lr
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.activation = activation
        self.model = None
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
    
    def fit(self,X,y):
        self.model= Net(X.shape[1],self.hidden_size,self.activation)
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).long()

        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay,momentum=self.momentum)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            print(torch.argmax(outputs,dim=1)[:10])
    
    def predict(self,X):
        if self.model is None:
            raise Exception('Model not trained')
        X = torch.from_numpy(X).float()
        outputs = self.model(X).detach().cpu().numpy()
        return np.argmax(outputs,axis=1)
    
    def score(self,X,y):
        if self.model is None:
            raise Exception('Model not trained')
        out = self.predict(X)
        return np.mean(out==y)

        