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

SUM_FEATURES = ['Rainfall_day', 'Precipitation']

def load_station_data(file_paths: List[str], station_names: List[str]) -> Dict[str, pd.DataFrame]:
    """Load data from multiple station CSV files"""
    station_data = {}
    
    for file_path, station_name in zip(file_paths, station_names):
        print(f"Loading data for station: {station_name}")
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['station_name'] = station_name
        station_data[station_name] = df
        
        print(f"  - Records: {len(df)}")
        print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  - Water level range: {df['water_level'].min():.2f} to {df['water_level'].max():.2f}m")
    
    return station_data

def train_multi_station_model(model, train_loader, val_loader, num_epochs, 
                             learning_rate, device, patience=20, min_delta=0.0001, use_seasons=False):
    """Train the multi-station model"""
    criterion = nn.MSELoss()
    
    # optimizer = torch.optim.Adam(
    #     model.parameters(), 
    #     lr=learning_rate
    # )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4   # start here; search 1e-4–1e-2
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=10,  # Different from early stopping patience
        factor=0.5,  # Less aggressive reduction
        min_lr=1e-7,  # Lower minimum
    )
    
    train_losses = []
    val_losses = []
    

    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_data in train_loader:
            if use_seasons:
                sequences, targets, station_ids, season_ids = batch_data
                season_ids = season_ids.to(device)
            else:
                sequences, targets, station_ids = batch_data
                season_ids = None

            sequences = sequences.to(device)
            targets = targets.to(device)
            station_ids = station_ids.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences, station_ids, season_ids)
            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                if use_seasons:
                    sequences, targets, station_ids, season_ids = batch_data
                    season_ids = season_ids.to(device)
                else:
                    sequences, targets, station_ids = batch_data
                    season_ids = None

                sequences = sequences.to(device)
                targets = targets.to(device)
                station_ids = station_ids.to(device)
                
                outputs = model(sequences, station_ids, season_ids)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}, Best Val Loss: {best_val_loss:.6f}')
        
        # Check if we should stop
        if epochs_without_improvement >= patience:

            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            print(f'Best validation loss: {best_val_loss:.6f}')
            # Restore best model
            model.load_state_dict(best_model_state)
            break
    
    return train_losses, val_losses

def train_lstm_optimized(
    model, train_loader, val_loader, config, device, num_epochs=100, patience=30
):
    """Optimized training function for LSTM models"""

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5, min_lr=1e-7
    )

    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_data in train_loader:
            if len(batch_data) == 4:
                sequences, targets, station_ids, season_ids = batch_data
                season_ids = season_ids.to(device)
            else:
                sequences, targets, station_ids = batch_data
                season_ids = None

            sequences = sequences.to(device)
            targets = targets.to(device)
            station_ids = station_ids.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(sequences, station_ids, season_ids)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 4:
                    sequences, targets, station_ids, season_ids = batch_data
                    season_ids = season_ids.to(device)
                else:
                    sequences, targets, station_ids = batch_data
                    season_ids = None

                sequences = sequences.to(device)
                targets = targets.to(device)
                station_ids = station_ids.to(device)

                outputs = model(sequences, station_ids, season_ids)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                val_batches += 1

        # Calculate average losses
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Print progress
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {avg_train_loss:.6f}, "
                f"Val Loss: {avg_val_loss:.6f}, "
                f"Best Val Loss: {best_val_loss:.6f}, "
                f"LR: {current_lr:.2e}"
            )

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            print(f"Best validation loss: {best_val_loss:.6f}")
            model.load_state_dict(best_model_state)
            break

    return model, train_losses, val_losses

def apply_station_scalers(df, feature_cols, station_scalers, feature_categories):
    """Apply existing station scalers to new data"""
    scaled_data = np.zeros((len(df), len(feature_cols)))
    
    # Map features to their positions
    feature_to_idx = {feat: idx for idx, feat in enumerate(feature_cols)}
    
    # Check if using old or new scaler format
    if 'features' in station_scalers:
        # Old format - single scaler for all features
        return station_scalers['features'].transform(df[feature_cols])
    
    # New format - different scalers for different types
    for feat_type, features in feature_categories.items():
        type_features = [f for f in features if f in feature_cols]
        
        if not type_features:
            continue
        
        if feat_type == 'cyclical':
            # Already normalized
            for feat in type_features:
                idx = feature_to_idx[feat]
                scaled_data[:, idx] = df[feat].values
                
        elif feat_type == 'rainfall' and 'rainfall' in station_scalers:
            # Apply log transform first
            temp_data = df[type_features].copy()
            for col in type_features:
                temp_data[col] = np.log1p(temp_data[col])
            
            # Then scale
            scaled_values = station_scalers['rainfall'].transform(temp_data)
            for i, feat in enumerate(type_features):
                idx = feature_to_idx[feat]
                scaled_data[:, idx] = scaled_values[:, i]
                
        elif feat_type in station_scalers:
            # Direct scaling
            scaled_values = station_scalers[feat_type].transform(df[type_features])
            for i, feat in enumerate(type_features):
                idx = feature_to_idx[feat]
                scaled_data[:, idx] = scaled_values[:, i]
    
    return scaled_data


def apply_global_scalers(df, feature_cols, global_scalers, feature_categories):
    """Apply global scalers to new data"""
    scaled_data = np.zeros((len(df), len(feature_cols)))
    
    # Map features to their positions
    feature_to_idx = {feat: idx for idx, feat in enumerate(feature_cols)}
    
    # Check if using old or new format
    if 'features' in global_scalers:
        # Old format
        return global_scalers['features'].transform(df[feature_cols])
    
    # New format with different scalers
    for feat_type, features in feature_categories.items():
        type_features = [f for f in features if f in feature_cols]
        
        if not type_features:
            continue
        
        if feat_type == 'cyclical':
            # Already normalized
            for feat in type_features:
                idx = feature_to_idx[feat]
                scaled_data[:, idx] = df[feat].values
                
        elif feat_type == 'rainfall':
            # Apply log transform first
            temp_data = df[type_features].copy()
            for col in type_features:
                temp_data[col] = np.log1p(temp_data[col])
            
            # Then scale
            scaled_values = global_scalers['rainfall'].transform(temp_data)
            for i, feat in enumerate(type_features):
                idx = feature_to_idx[feat]
                scaled_data[:, idx] = scaled_values[:, i]
                
        elif feat_type in ['pressure', 'humidity']:
            # MinMax scaling
            scaled_values = global_scalers['minmax'].transform(df[type_features])
            for i, feat in enumerate(type_features):
                idx = feature_to_idx[feat]
                scaled_data[:, idx] = scaled_values[:, i]
                
        else:
            # Standard scaling
            scaled_values = global_scalers['standard'].transform(df[type_features])
            for i, feat in enumerate(type_features):
                idx = feature_to_idx[feat]
                scaled_data[:, idx] = scaled_values[:, i]
    
    return scaled_data


def create_lagged_features(df, feature_set):
    for feature in feature_set:
        if feature in ['water_level', 'Season']: 
            continue
            
        col = f'{feature}_4day_lagged'
        if feature in SUM_FEATURES:
            df[col] = df[feature].rolling(4, min_periods=1).sum().shift(1)
        else:
            df[col] = df[feature].rolling(4, min_periods=1).mean().shift(1)
        df[col].fillna(df[feature].iloc[0], inplace=True)
    return df


def categorize_features(feature_set):
    """Categorize features for appropriate normalization"""
    categories = {
        'rainfall': [],
        'pressure': [],
        'temperature': [],
        'humidity': [],
        'wind': [],
        'sun': [],
        'water_level': [],
        'cyclical': [],
        'other': []
    }
    
    for feature in feature_set:
        feature_lower = feature.lower()
        
        if 'rain' in feature_lower:
            categories['rainfall'].append(feature)
        elif 'pressure' in feature_lower:
            categories['pressure'].append(feature)
        elif 'temp' in feature_lower:
            categories['temperature'].append(feature)
        elif 'humid' in feature_lower:
            categories['humidity'].append(feature)
        elif 'wind' in feature_lower:
            categories['wind'].append(feature)
        elif 'sun' in feature_lower:
            categories['sun'].append(feature)
        elif 'water_level' in feature_lower:
            categories['water_level'].append(feature)
        elif feature == 'Season':
            categories['cyclical'].append(feature)
        else:
            categories['other'].append(feature)
    
    return categories


def process_features(df, feature_set):
    """Process features including Season conversion"""
    df_processed = df.copy()
    feature_cols = []
    
    for feature in feature_set:
        if feature == 'Season':
            # Convert to cyclical encoding
            df_processed['season_sin'] = np.sin(2 * np.pi * df['Season'] / 4)
            df_processed['season_cos'] = np.cos(2 * np.pi * df['Season'] / 4)
            feature_cols.extend(['season_sin', 'season_cos'])
        else:
            feature_cols.append(feature)
    
    return df_processed, feature_cols


def smart_fillna(df, feature_categories):
    """Intelligently fill NAs based on feature type"""
    # Rainfall features - fill with 0
    for feature in feature_categories['rainfall']:
        if feature in df.columns:
            df[feature] = df[feature].fillna(0)
    
    # Smooth meteorological variables - interpolate
    smooth_categories = ['pressure', 'temperature', 'humidity', 'wind', 'sun']
    for category in smooth_categories:
        for feature in feature_categories[category]:
            if feature in df.columns:
                df[feature] = df[feature].interpolate(method='linear', limit_direction='both')
    
    # Water level - forward fill
    for feature in feature_categories['water_level']:
        if feature in df.columns:
            df[feature] = df[feature].fillna(method='ffill').fillna(method='bfill')
    
    # Any remaining - backward then forward fill
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df


def scale_features_per_station(df, feature_cols, feature_categories):
    """Scale features with appropriate methods per station"""
    scaled_data = df[feature_cols].copy()
    scalers = {}
    
    # Process each feature type
    for feat_type, features in feature_categories.items():
        type_features = [f for f in features if f in feature_cols]
        
        if not type_features:
            continue
            
        if feat_type == 'cyclical' or 'season_sin' in feature_cols:
            # Cyclical features are already normalized
            continue
            
        elif feat_type == 'rainfall':
            # Log transform + RobustScaler for rainfall
            scaler = RobustScaler()
            for col in type_features:
                # Apply log(1+x) transform
                scaled_data[col] = np.log1p(df[col])
            # Then scale
            scaled_data[type_features] = scaler.fit_transform(scaled_data[type_features])
            scalers['rainfall'] = scaler
            
        elif feat_type in ['pressure', 'humidity']:
            # MinMaxScaler for bounded features
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_data[type_features] = scaler.fit_transform(df[type_features])
            scalers[feat_type] = scaler
            
        else:
            # StandardScaler for others
            scaler = StandardScaler()
            scaled_data[type_features] = scaler.fit_transform(df[type_features])
            scalers[feat_type] = scaler
    
    return scaled_data.values, scalers


def scale_features_global(df, feature_cols, feature_categories, global_scalers):
    """Scale features using global scalers"""
    scaled_data = df[feature_cols].copy()
    
    for feat_type, features in feature_categories.items():
        type_features = [f for f in features if f in feature_cols]
        
        if not type_features:
            continue
            
        if feat_type == 'cyclical' or 'season_sin' in feature_cols:
            continue
            
        elif feat_type == 'rainfall':
            # Apply log transform first
            for col in type_features:
                scaled_data[col] = np.log1p(df[col])
            # Then scale
            scaled_data[type_features] = global_scalers['rainfall'].transform(scaled_data[type_features])
            
        elif feat_type in ['pressure', 'humidity']:
            scaled_data[type_features] = global_scalers['minmax'].transform(df[type_features])
            
        else:
            scaled_data[type_features] = global_scalers['standard'].transform(df[type_features])
    
    return scaled_data.values


# Helper function for inverse transformation (useful for predictions)
def inverse_transform_predictions(predictions, scalers, station_name=None):
    """
    Inverse transform predictions back to original scale
    
    Args:
        predictions: Scaled predictions
        scalers: Dictionary of scalers
        station_name: Name of station (for per-station normalization)
    
    Returns:
        Original scale predictions
    """
    if 'global' in scalers:
        # Global normalization
        target_scaler = scalers['global']['target']
    else:
        # Per-station normalization
        if station_name is None:
            raise ValueError("station_name required for per-station normalization")
        target_scaler = scalers[station_name]['target']
    
    # Reshape if needed
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    
    # Inverse transform
    original_scale = target_scaler.inverse_transform(predictions)
    
    return original_scale



def prepare_multi_station_data(station_data: Dict[str, pd.DataFrame], 
                              feature_set,
                              sequence_length: int = 30, 
                              prediction_days: int = 7,
                              normalize_per_station: bool = True,
                              last_n_days = 0,
                              date_range_days: tuple = None,
                              use_seasons: bool = False):
    """Prepare data from multiple stations for training with proper normalization"""
    
    all_sequences = []
    all_targets = []
    all_station_ids = []
    all_season_ids = [] if use_seasons else None

    # Validate parameters
    if last_n_days > 0 and date_range_days is not None:
        raise ValueError("Cannot specify both last_n_days and date_range_days. Choose one.")
    
    # Reduce data if needed
    station_data_reduced = {}
    for station_name, df in station_data.items():
        df_copy = df.copy()  # Work with a copy to avoid modifying original
        
        if date_range_days is not None:
            # Handle date range selection
            start_idx, end_idx = date_range_days
            total_days = len(df_copy)
            
            # Convert negative indices to positive
            if start_idx < 0:
                start_idx = total_days + start_idx
            if end_idx <= 0:
                end_idx = total_days + end_idx
            
            # Ensure valid range
            start_idx = max(0, start_idx)
            if(start_idx == 0):
                start_idx = 1 - total_days
            end_idx = min(total_days, end_idx)
            
            if start_idx >= end_idx:
                raise ValueError(f"Invalid date range: start {start_idx}({date_range_days[0]}) >= end {end_idx}({date_range_days[1]})")
            
            df_copy = df_copy.iloc[start_idx:end_idx]
            print(f"Station {station_name}: Selected days {start_idx} to {end_idx} "
                  f"({end_idx - start_idx} days)")
            
        elif last_n_days > 0:
            # Original behavior - take last n days
            df_copy = df_copy.tail(last_n_days)
            print(f"Station {station_name}: Using last {last_n_days} days")
        
        station_data_reduced[station_name] = df_copy
    
    # Create station name to ID mapping
    station_to_id = {name: idx for idx, name in enumerate(station_data.keys())}
    
    # Categorize features for different normalization strategies
    feature_categories = categorize_features(feature_set)
    
    # Initialize scalers
    if normalize_per_station:
        station_scalers = {}
    else:
        # Initialize different scalers for different feature types
        global_scalers = {
            'rainfall': RobustScaler(),
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(feature_range=(-1, 1)),
            'target': StandardScaler()
        }
        
        # Fit global scalers on all data
        all_features_by_type = {
            'rainfall': [],
            'standard': [],
            'minmax': []
        }
        all_targets_data = []
        
        # First pass: collect all data for fitting scalers
        for station_name, df in station_data_reduced.items():
            # Process features and handle Season
            df_processed, feature_cols = process_features(df, feature_set)
            
            # Separate features by type
            for feat_type, features in feature_categories.items():
                if feat_type == 'cyclical':
                    continue  # Already normalized by sin/cos
                
                type_features = [f for f in features if f in feature_cols]
                if type_features:
                    if feat_type == 'rainfall':
                        # Apply log transform before collecting
                        rain_data = df_processed[type_features].values
                        rain_data_log = np.log1p(rain_data)  # log(1+x)
                        all_features_by_type['rainfall'].append(rain_data_log)
                    elif feat_type in ['pressure', 'humidity']:
                        all_features_by_type['minmax'].append(df_processed[type_features].values)
                    else:
                        all_features_by_type['standard'].append(df_processed[type_features].values)
            
            all_targets_data.append(df_processed[['water_level']].values)
        
        # Fit scalers
        if all_features_by_type['rainfall']:
            global_scalers['rainfall'].fit(np.vstack(all_features_by_type['rainfall']))
        if all_features_by_type['standard']:
            global_scalers['standard'].fit(np.vstack(all_features_by_type['standard']))
        if all_features_by_type['minmax']:
            global_scalers['minmax'].fit(np.vstack(all_features_by_type['minmax']))
        
        global_scalers['target'].fit(np.vstack(all_targets_data))
    
    # Process each station
    for station_name, df in station_data_reduced.items():
        print(f"\nProcessing station: {station_name}")
        
        # Extract seasons if needed
        seasons = None
        if use_seasons:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            seasons = ((df.index.month % 12 + 3) // 3 - 1).values
        
        # Process features
        df_processed, feature_cols = process_features(df, feature_set)
        
        # Handle missing values intelligently
        df_processed = smart_fillna(df_processed, feature_categories)
        
        # Save processed data
        # df_processed.to_csv(f"{exp['csv_output_path']}{station_name}_processed.csv", index=False)
        
        # Scale the data
        if normalize_per_station:
            scaled_features, station_scaler_dict = scale_features_per_station(
                df_processed, feature_cols, feature_categories
            )
            
            # Scale target
            scaler_target = StandardScaler()
            scaled_target = scaler_target.fit_transform(df_processed[['water_level']])
            
            station_scalers[station_name] = {
                **station_scaler_dict,
                'target': scaler_target,
                'feature_cols': feature_cols,  # Store feature column order
                'feature_categories': feature_categories  # Store categories for inverse transform
            }
        else:
            # Use global normalization
            scaled_features = scale_features_global(
                df_processed, feature_cols, feature_categories, global_scalers
            )
            scaled_target = global_scalers['target'].transform(df_processed[['water_level']])
        
        # Create sequences for this station
        station_id = station_to_id[station_name]
        
        for i in range(len(scaled_features) - sequence_length - prediction_days + 1):
            seq = scaled_features[i:i + sequence_length]
            target = scaled_target[i + sequence_length:i + sequence_length + prediction_days]
            
            all_sequences.append(seq)
            all_targets.append(target.flatten())
            all_station_ids.append(station_id)
            
            if use_seasons:
                season_id = seasons[i + sequence_length]
                all_season_ids.append(season_id)
    
    # Convert to numpy arrays
    all_sequences = np.array(all_sequences)
    all_targets = np.array(all_targets)
    all_station_ids = np.array(all_station_ids)
    if use_seasons:
        all_season_ids = np.array(all_season_ids)
    
    # Prepare scalers for return
    if normalize_per_station:
        scalers = station_scalers
    else:
        scalers = {
            'global': {
                **global_scalers,
                'feature_cols': feature_cols,
                'feature_categories': feature_categories
            }
        }
    
    # Return based on configuration
    if use_seasons:
        return all_sequences, all_targets, all_station_ids, all_season_ids, station_to_id, scalers
    else:
        return all_sequences, all_targets, all_station_ids, station_to_id, scalers


def test_on_full_timeline(
    model,
    feature_set,
    test_station_data,
    station_name,
    station_to_id,
    scalers,
    sequence_length,
    prediction_days,
    device,
    normalize_per_station=True,
    stride=1,
    use_seasons=False,
    start_date=None,
    end_date=None,
    save_metrics_path=None,
    anchor_predictions=True,  # New parameter to control anchoring
    anchoring_method="replace",  # Options: "replace", "adjust", "blend"
):
    """
    Test the model on a station by making rolling predictions across the entire timeline.

    This function creates predictions for every possible window in the timeline, allowing
    for comprehensive evaluation of model performance over time.

    Parameters:
    -----------
    model : torch.nn.Module
        Trained model
    feature_set : list
        List of features to use
    test_station_data : pd.DataFrame
        Test station data
    station_name : str
        Name of the test station
    station_to_id : dict
        Mapping of station names to IDs
    scalers : dict
        Scalers from training
    sequence_length : int
        Length of input sequences
    prediction_days : int
        Number of days to predict ahead
    device : torch.device
        Device to run on
    normalize_per_station : bool
        Whether to normalize per station
    stride : int
        Step size for rolling window (default=1 for daily predictions)
    use_seasons : bool
        Whether to use seasonal embeddings
    start_date : str or pd.Timestamp, optional
        Start date for predictions (format: 'YYYY-MM-DD')
    end_date : str or pd.Timestamp, optional
        End date for predictions (format: 'YYYY-MM-DD')
    save_metrics_path : str, optional
        Path to save metrics (without extension, will save as .json and .csv)
    anchor_predictions : bool
        Whether to anchor predictions to actual value at t=0 (default=True)
    anchoring_method : str
        Method for anchoring: "replace", "adjust", or "blend" (default="replace")

    Returns:
    --------
    dict : Dictionary containing:
        - 'predictions': Array of all predictions
        - 'actual_values': Array of actual values
        - 'dates': Array of prediction start dates
        - 'station_name': Name of the station
        - 'metrics': Dictionary of evaluation metrics
        - 'prediction_windows': List of (start_idx, end_idx) tuples for each prediction
    """
    model.eval()

    # Prepare test data
    df = test_station_data.copy()

    # Extract seasons before any data reduction
    seasons = None
    if use_seasons:
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df.index = pd.to_datetime(df["date"])
            else:
                df.index = pd.to_datetime(df.index)
        seasons = ((df.index.month % 12 + 3) // 3 - 1).values

    # Apply date filtering if specified
    if start_date is not None or end_date is not None:
        if "date" in df.columns:
            date_col = pd.to_datetime(df["date"])
        else:
            date_col = df.index

        mask = pd.Series([True] * len(df))
        if start_date is not None:
            mask &= date_col >= pd.to_datetime(start_date)
        if end_date is not None:
            mask &= date_col <= pd.to_datetime(end_date)

        df = df[mask]
        if use_seasons:
            seasons = seasons[mask]

        print(f"Date range filtered: {date_col[mask].min()} to {date_col[mask].max()}")

    # Process features
    df_processed, feature_cols = process_features(df, feature_set)
    feature_categories = categorize_features(feature_cols)

    # Handle missing values
    df_processed = smart_fillna(df_processed, feature_categories)

    # Scale features
    if normalize_per_station:
        if station_name in scalers:
            scaled_features = apply_station_scalers(
                df_processed, feature_cols, scalers[station_name], feature_categories
            )
            scaler_target = scalers[station_name]["target"]
        else:
            print(f"Warning: {station_name} not in training set. Creating new scalers.")
            scaled_features, new_scalers = scale_features_per_station(
                df_processed, feature_cols, feature_categories
            )
            scaler_target = StandardScaler()
            scaler_target.fit(df_processed[["water_level"]])
    else:
        scaled_features = apply_global_scalers(
            df_processed, feature_cols, scalers["global"], feature_categories
        )
        scaler_target = scalers["global"]["target"]

    # Get station ID
    station_id = station_to_id.get(station_name, 0)

    # Initialize storage for results
    all_predictions = []
    all_actuals = []
    all_dates = []
    prediction_windows = []

    # Calculate the total number of possible predictions
    total_predictions = (
        len(df_processed) - sequence_length - prediction_days + 1
    ) // stride

    if total_predictions <= 0:
        print(
            f"Insufficient data for predictions. Need at least {sequence_length + prediction_days} days, have {len(df_processed)}"
        )
        return {
            "predictions": np.array([]),
            "actual_values": np.array([]),
            "dates": np.array([]),
            "station_name": station_name,
            "metrics": {
                "MAE_overall": np.nan,
                "RMSE_overall": np.nan,
                "MAPE_overall": np.nan,
                "R2_overall": np.nan,
            },
            "prediction_windows": [],
        }

    print(f"\nMaking {total_predictions} predictions for station {station_name}")
    print(f"Timeline: {df_processed.index[0]} to {df_processed.index[-1]}")
    if anchor_predictions:
        print(f"Anchoring method: {anchoring_method}")

    # Make predictions across the timeline
    with torch.no_grad():
        for i in range(
            0, len(scaled_features) - sequence_length - prediction_days + 1, stride
        ):
            # Get sequence
            sequence = scaled_features[i : i + sequence_length]

            if len(sequence) != sequence_length:
                continue

            # Prepare tensors
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            station_id_tensor = torch.LongTensor([station_id]).to(device)

            # Handle season if needed
            season_id_tensor = None
            if use_seasons:
                season_idx = i + sequence_length
                if 0 <= season_idx < len(seasons):
                    season_id = seasons[season_idx]
                    season_id_tensor = torch.LongTensor([season_id]).to(device)

            # Make prediction
            prediction_scaled = (
                model(sequence_tensor, station_id_tensor, season_id_tensor)
                .cpu()
                .numpy()
            )

            # Reshape and inverse transform
            prediction_scaled = prediction_scaled.reshape(-1, prediction_days)
            prediction = scaler_target.inverse_transform(prediction_scaled).flatten()

            # Get actual values
            actual_start_idx = i + sequence_length
            actual_end_idx = actual_start_idx + prediction_days
            actual_values = df_processed.iloc[actual_start_idx:actual_end_idx][
                "water_level"
            ].values

            # Apply anchoring if enabled
            if anchor_predictions and len(actual_values) > 0:
                current_actual = actual_values[0]  # Actual value at t=0

                if anchoring_method == "replace":
                    # Simply replace first prediction with actual value
                    prediction[0] = current_actual

                elif anchoring_method == "adjust":
                    # Shift entire prediction sequence by the difference
                    adjustment = current_actual - prediction[0]
                    prediction = prediction + adjustment

                elif anchoring_method == "blend":
                    # Gradually blend from actual to predicted
                    # This maintains continuity while allowing the model's dynamics to emerge
                    blend_weights = np.exp(
                        -np.arange(prediction_days) / (prediction_days / 3)
                    )
                    adjustment = current_actual - prediction[0]
                    prediction = prediction + adjustment * blend_weights

            # Get dates
            if "date" in df_processed.columns:
                pred_date = df_processed.iloc[actual_start_idx]["date"]
            else:
                pred_date = df_processed.iloc[actual_start_idx].name

            # Store results
            all_predictions.append(prediction)
            all_actuals.append(actual_values)
            all_dates.append(pred_date)
            prediction_windows.append((actual_start_idx, actual_end_idx))

            # Progress update every 100 predictions
            if (len(all_predictions) % 100) == 0:
                print(
                    f"  Processed {len(all_predictions)}/{total_predictions} predictions..."
                )

    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    all_dates = np.array(all_dates)

    # Calculate metrics
    metrics = calculate_prediction_metrics(
        all_predictions, all_actuals, prediction_days
    )

    print(f"\nCompleted {len(all_predictions)} predictions")
    print(f"Overall metrics:")
    print(f"  MAE: {metrics['MAE_overall']:.4f}")
    print(f"  RMSE: {metrics['RMSE_overall']:.4f}")
    print(f"  MAPE: {metrics['MAPE_overall']:.2f}%")
    print(f"  R²: {metrics['R2_overall']:.4f}")

    # Save metrics if path is provided
    if save_metrics_path:
        save_evaluation_metrics(
            metrics,
            all_predictions,
            all_actuals,
            all_dates,
            station_name,
            save_metrics_path,
        )

    return {
        "predictions": all_predictions,
        "actual_values": all_actuals,
        "dates": all_dates,
        "station_name": station_name,
        "metrics": metrics,
        "prediction_windows": prediction_windows,
    }

