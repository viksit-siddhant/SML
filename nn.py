import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

class Dataset(torch.utils.data.Dataset):
    def __init__(self,x,y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

def train(x,y, validation_split = 0.1, epochs = 100):
    model = Net(x.shape[1], np.amax(y)+1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=0.00005,lr = 0.001)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=validation_split)

    train_dataset = torch.utils.data.DataLoader(Dataset(x_train,y_train),batch_size=32,shuffle=True)
    test_dataset = torch.utils.data.DataLoader(Dataset(x_test,y_test),batch_size=32,shuffle=True)

    for epoch in range(epochs):
        num_correct = 0
        num_samples = 0
        for x_train, y_train in train_dataset:
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            num_correct += (outputs.argmax(1) == y_train).sum()
            num_samples += outputs.shape[0]
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {num_correct/num_samples:.4f}')
        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for x_test, y_test in test_dataset:
                outputs = model(x_test)
                num_correct += (outputs.argmax(1) == y_test).sum()
                num_samples += outputs.shape[0]
            print(f'Validation Accuracy: {num_correct/num_samples:.4f}')
    return model