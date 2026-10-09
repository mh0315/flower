import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms,datasets
import os
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torch import optim
from sklearn.metrics import accuracy_score



data_dir = './flowers/'
train_dir = data_dir + '/train'
test_dir = data_dir + '/test'
EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp') 
for folder in [train_dir, test_dir]: # 找出坏图
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(EXTS):
             path = os.path.join(root, file)
            try:
                with Image.open(path) as img:
                    img.verify()  
            except Exception:
              os.remove(path)
train_transforms = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
    ])
test_transforms = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor()
    ])

train_dataset = datasets.ImageFolder( root='./flowers/train', transform=train_transforms)
test_dataset = datasets.ImageFolder( root='./flowers/test', transform=test_transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2,drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2,drop_last=False)
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 =nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=1)

        self.conv2 =nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=2, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=1)
        
        self.fc = nn.Linear(in_features=64*27*27, out_features=5)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu(x)
        
        x = x.view(x.shape[0],-1) #展平
        x = self.fc(x)
        
        return x
    
def main(epoches):
    model = cnn()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epoches):
        Loss = 0
        for x,y in train_loader:
            outputs = model(x)
            loss = criterion(outputs,y)
            loss  += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           

    model.eval()
    with torch.no_grad():
        for x,y in test_loader:
            outputs = model.forward(x=x)
            pre_y = torch.argmax(outputs,dim=1)
            acc =accuracy_score(pre_y,y)
            return acc
if __name__ == '__main__':
    acc = main(epoches=10)
    print(acc)

