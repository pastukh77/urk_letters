from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2),                              
            nn.ReLU(),                      
            nn.MaxPool2d(2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(8, 16, kernel_size=5, padding=2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )

        self.fc1 = nn.Linear(1296, 64)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(64, 32)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

