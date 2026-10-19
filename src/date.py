import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision import transforms,datasets
import os
from PIL import Image
from torch.utils.data import DataLoader
from torch import optim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoches",type=int,help="choose epoches")
args = parser.parse_args()

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
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_dataset = datasets.ImageFolder( root='./flowers/train', transform=train_transforms)
test_dataset = datasets.ImageFolder( root='./flowers/test', transform=test_transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0,drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0,drop_last=False)
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.conv1 =nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11,11), stride=4,padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3),stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.pool2 =nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 =nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 =nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        
        self.conv5 =nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.pool3 =nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 =nn.Linear(256 * 6 * 6, 4096)
        self.fc2 =nn.Linear(4096, 4096)
        self.fc3 =nn.Linear(4096, 18)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.pool3(x)
        x = self.relu(x)
        
        x = x.view(x.shape[0],-1) 
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x
    
def main(epoches):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Mymodel().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_acc_list = []
    test_acc_list = []
    for epoch in range(epoches):
        model.train()
        train_correct = 0
        train_total = 0
        Loss = 0
        for x,y in train_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs,y)
            Loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == y).sum().item()
            train_total += len(y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc = train_correct / train_total
        train_acc_list.append(train_acc)

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x,y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == y).sum().item()
                test_total += len(y)
        test_acc = test_correct / test_total   
        test_acc_list.append(test_acc)    
    print(f"| Train Acc: {train_acc:.4f} |"
          f"| Test Acc: {test_acc:.4f}|")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epoches+1), train_acc_list, label='Train Accuracy', marker='o')
    plt.plot(range(1, epoches+1), test_acc_list, label='Test Accuracy', marker='s')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main(args.epoches)
