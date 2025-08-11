import torch
import numpy as np

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, tabs, labels in dataloader:
        imgs, tabs, labels = imgs.to(device), tabs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs, tabs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(dataloader.dataset)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for imgs, tabs, labels in dataloader:
            imgs, tabs, labels = imgs.to(device), tabs.to(device), labels.to(device)
            outputs = model(imgs, tabs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
    return running_loss / len(dataloader.dataset)

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
