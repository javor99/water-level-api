import torch
import torch.nn as nn
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
import warnings
import math

warnings.filterwarnings("ignore")


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class BaseModel(nn.Module):
    """Base class for all models to ensure consistent interface"""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        num_stations,
        station_embedding_dim=8,
        use_seasons=False,
        season_embedding_dim=4,
        dropout=0.5,
    ):
        super(BaseModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_stations = num_stations
        self.station_embedding_dim = station_embedding_dim
        self.use_seasons = use_seasons
        self.season_embedding_dim = season_embedding_dim
        self.dropout_rate = dropout

        # Station and season embeddings (shared across models)
        self.station_embedding = nn.Embedding(num_stations, station_embedding_dim)
        if self.use_seasons:
            self.season_embedding = nn.Embedding(4, season_embedding_dim)
            self.total_input_size = (
                input_size + station_embedding_dim + season_embedding_dim
            )
        else:
            self.total_input_size = input_size + station_embedding_dim

    def get_embeddings(self, x, station_ids, season_ids):
        """Get station and season embeddings"""
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Station embeddings
        station_embed = self.station_embedding(station_ids)
        station_embed = station_embed.unsqueeze(1).expand(batch_size, seq_len, -1)

        # Season embeddings if enabled
        if self.use_seasons:
            if season_ids is None:
                raise ValueError("season_ids must be provided when use_seasons=True")
            season_embed = self.season_embedding(season_ids)
            season_embed = season_embed.unsqueeze(1).expand(batch_size, seq_len, -1)
            x = torch.cat([x, station_embed, season_embed], dim=2)
        else:
            x = torch.cat([x, station_embed], dim=2)

        return x


# 1. TRANSFORMER MODEL
class MultiStationTransformer(BaseModel):
    """Transformer model with station embeddings"""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        num_stations,
        station_embedding_dim=8,
        use_seasons=False,
        season_embedding_dim=4,
        dropout=0.5,
    ):
        super().__init__(
            input_size,
            hidden_size,
            num_layers,
            output_size,
            num_stations,
            station_embedding_dim,
            use_seasons,
            season_embedding_dim,
            dropout,
        )

        # Input projection
        self.input_projection = nn.Linear(self.total_input_size, hidden_size)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, station_ids, season_ids=None):
        # Get embeddings
        x = self.get_embeddings(x, station_ids, season_ids)

        # Project to hidden size
        x = self.input_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer encoding
        x = self.transformer(x)

        # Use the last timestep output
        x = x[:, -1, :]

        # Output layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# 3. GRU MODEL
class MultiStationGRU(BaseModel):
    """GRU model with station embeddings"""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        num_stations,
        station_embedding_dim=8,
        use_seasons=False,
        season_embedding_dim=4,
        dropout=0.5,
    ):
        super().__init__(
            input_size,
            hidden_size,
            num_layers,
            output_size,
            num_stations,
            station_embedding_dim,
            use_seasons,
            season_embedding_dim,
            dropout,
        )

        # GRU layer
        self.gru = nn.GRU(
            self.total_input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, station_ids, season_ids=None):
        # Get embeddings
        x = self.get_embeddings(x, station_ids, season_ids)

        # GRU forward pass
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)
        out = out[:, -1, :]

        # Output layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out




class LSTMConfig:
    """Configuration for LSTM models"""

    def __init__(
        self,
        input_size,
        hidden_size=256,
        num_layers=3,
        output_size=7,
        num_stations=10,
        station_embedding_dim=8,
        use_seasons=False,
        season_embedding_dim=4,
        dropout=0.5,
        learning_rate=0.0001,
        weight_decay=1e-4,
        gradient_clip=1.0,
        batch_norm=True,
        layer_norm=True,
        bidirectional=False,
        attention=False,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_stations = num_stations
        self.station_embedding_dim = station_embedding_dim
        self.use_seasons = use_seasons
        self.season_embedding_dim = season_embedding_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.bidirectional = bidirectional
        self.attention = attention




# Model factory function
def get_model(
    model_name,
    input_size,
    hidden_size,
    num_layers,
    output_size,
    num_stations,
    station_embedding_dim=8,
    use_seasons=False,
    season_embedding_dim=4,
    dropout=0.5,
):
    """Factory function to get model by name"""

    models = {
        "transformer": MultiStationTransformer,
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(models.keys())}"
        )

    return models[model_name](
        input_size,
        hidden_size,
        num_layers,
        output_size,
        num_stations,
        station_embedding_dim,
        use_seasons,
        season_embedding_dim,
        dropout,
    )
