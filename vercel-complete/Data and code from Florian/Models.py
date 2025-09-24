import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff()
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from torch.utils.data import Dataset, DataLoader
import os
import json
from typing import Dict, List, Tuple
import warnings
import datetime
warnings.filterwarnings('ignore')
import seaborn as sns
from config_loader import load_config, load_mappings

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

class MultiStationDataset(Dataset):
    """Dataset for multiple water stations"""
    def __init__(self, sequences, targets, station_ids, season_ids=None):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.station_ids = torch.LongTensor(station_ids)
        self.season_ids = torch.LongTensor(season_ids) if season_ids is not None else None
        self.use_seasons = season_ids is not None
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.use_seasons:
            return self.sequences[idx], self.targets[idx], self.station_ids[idx], self.season_ids[idx]
        else:
            return self.sequences[idx], self.targets[idx], self.station_ids[idx]

class MultiStationLSTM(nn.Module):
    """LSTM model with station embeddings"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 num_stations, station_embedding_dim=8, use_seasons=False, season_embedding_dim=4, dropout=0.5):
        super(MultiStationLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_seasons = use_seasons
        
        # Station embedding layer
        self.station_embedding = nn.Embedding(num_stations, station_embedding_dim)

        # Season embedding layer (only if use_seasons is True)
        if self.use_seasons:
            self.season_embedding = nn.Embedding(4, season_embedding_dim)
            total_input_size = input_size + station_embedding_dim + season_embedding_dim
        else:
            total_input_size = input_size + station_embedding_dim
        
        # LSTM takes input features + embedding
        self.lstm = nn.LSTM(total_input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, station_ids, season_ids):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Get station embeddings
        station_embed = self.station_embedding(station_ids)  # (batch_size, embedding_dim)
        station_embed = station_embed.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # Handle season embeddings if enabled
        if self.use_seasons:
            if season_ids is None:
                raise ValueError("season_ids must be provided when use_seasons=True")
            season_embed = self.season_embedding(season_ids)
            season_embed = season_embed.unsqueeze(1).expand(batch_size, seq_len, -1)
            # Concatenate all embeddings
            x = torch.cat([x, station_embed, season_embed], dim=2)
        else:
            # Only concatenate station embeddings
            x = torch.cat([x, station_embed], dim=2)
        
        # LSTM forward pass
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out