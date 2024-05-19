import pandas as pd 
from data import create_stock_table_if_not_exists
from data.loader import prepare_data, create_data_loaders
from model.lstm import LSTMModel
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn   
import torch.optim as optim

# Hyperparameters
input_size = 13
hidden_size = 64
num_layers = 2
output_size = 1
num_epochs = 100
learning_rate = 0.001
seq_length = 30
batch_size = 64

def _train_model(model, data_loader, criterion, optimizer, num_epochs, device):
    model.train()
    loss = None
    for epoch in range(num_epochs):
        for sequences, targets in data_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item() if loss else None:.4f}')
        torch.save(model.state_dict(), 'lstm_model.pth')


def train(data):
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Checking if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    sequences_tensor, targets_tensor = prepare_data(data, seq_length) 
    data_loader = create_data_loaders(sequences_tensor, targets_tensor, batch_size)
    _train_model(model, data_loader, criterion, optimizer, num_epochs, device)



def evaluate(data):
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load('lstm_model.pth'))
    model.eval()
    sequences_tensor, targets_tensor = prepare_data(data, seq_length) 
    data_loader = create_data_loaders(sequences_tensor, targets_tensor, batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        total_loss = 0
        criterion = nn.MSELoss()
        for sequences, targets in data_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print(f'Average Loss: {avg_loss:.4f}')




