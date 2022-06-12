from PIL import Image
import torch
import numpy as np
from cnn import CNN
from data import train_loader, test_loader
from sklearn.metrics import accuracy_score
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs")


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
model = CNN()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)   
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

model.to(device)

epochs = 100

continue_training = False

if continue_training:
    checkpoint = torch.load("cnn.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

criterion = nn.CrossEntropyLoss()

total_train_loss = []

def train(epochs):
    for epoch in range(epochs):

        model.train()
        start_time = time.time()
        train_loss = []
        for batch, labels in tqdm(train_loader):
            out = model(batch)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        lr_scheduler.step()    
        
        epoch_train_loss = np.mean(train_loss)
        total_train_loss.append(epoch_train_loss)
        print(f'Epoch №{epoch} train loss is {epoch_train_loss:.4f}')

        
        time_elapsed = time.time() - start_time
        print("Time elapsed: ",time_elapsed)
        
        torch.save({
                'epoch': epoch,
                'loss': epoch_train_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, "cnn.pth")

        plt.plot(total_train_loss)
        writer.add_scalar("train_loss", torch.tensor(epoch_train_loss), epoch)
        plt.savefig("loss.png")

if __name__ == '__main__':
    train(epochs)