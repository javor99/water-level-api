from importlib import reload
from re import A
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import torch
from torch.utils.data import  DataLoader
import os
import json
from typing import List
import warnings
from datetime import datetime
import Models_Alt
from config_loader import load_config, load_mappings
import Models, Trainer, Plotter, Analyzers, Metrics
import signal
import sys
import gc
import Station_Train_Test_Setup
warnings.filterwarnings('ignore')
plt.ioff()


changed_dir_times = 0

if changed_dir_times == 0:
    absolute_base_path = os.getcwd()

os.chdir(absolute_base_path)

# Global flag for graceful shutdown
stop_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global stop_requested
    print("\n\n⚠️  Stop requested! Will stop after current experiment completes...")
    print("Press Ctrl+C again to force quit (not recommended)")
    stop_requested = True

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def check_stop_file():
    """Check if stop file exists"""
    return os.path.exists('STOP_EXPERIMENTS.txt')

def load_progress():
    """Load progress from checkpoint file"""
    progress_file = 'results/experiments/experiment_progress.json'
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {
        'completed_experiments': [],
        'last_completed': None,
        'total_experiments': 0,
        'start_time': str(datetime.now())
    }

def save_progress(progress):
    """Save progress to checkpoint file"""
    progress_file = 'results/experiments/experiment_progress.json'
    os.makedirs('results/experiments', exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
        
        
def save_metrics(test_station, test_station_n, metrics):
    metric_file = f"{absolute_base_path}/results/experiments/{test_station_n:02d}_{test_station}/metrics.json"
    with open(metric_file, 'w') as f:
        json.dump(metrics, f, indent=2)
        
        
def get_experiment_id(exp_idx, exp_name, test_station):
    """Generate unique experiment ID"""
    return f"{exp_idx}_{exp_name}_{test_station}"


def save_experiment_info(experiment_id, experiment_name, exp, test_station, test_station_n):
    """Save experiment configuration"""
    
    # exp_dir = f"experiments/{experiment_id}_{experiment_name}/{test_station_n}_{test_station}"
    # date_range_train = exp['date_range_train']
    
    # if date_range_train is None:
    #     date_range_str = ''
    # else:
    #     date_range_str = f"{abs(date_range_train[0])}_days"
    
    exp_dir = f"results/experiments/{test_station_n:02d}_{test_station}/{experiment_id:03d}_{experiment_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config as JSON
    with open(f"{exp_dir}/config.json", 'w') as f:
        json.dump(exp, f, indent=2)
    
    # Save human-readable version
    with open(f"{exp_dir}/config.txt", 'w') as f:
        f.write(f"Experiment ID: {experiment_id}\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Test Station: {test_station}\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write("="*50 + "\n\n")
        f.write("PARAMETERS:\n")
        f.write("-"*30 + "\n")
        for key, value in sorted(exp.items()):
            if isinstance(value, list):
                f.write(f"{key}:\n")
                for item in value:
                    f.write(f"  - {item}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    return exp_dir



# Main execution function
def main(
    model_type, 
    station_files: List[str],
    station_names: List[str],
    features,
    train_stations: List[str],
    test_station_name: str = None,
    normalize_per_station: bool = True,
    existing_model=None,
    use_seasons: bool = False,
    use_old: bool = False,
    patience: int = 20,
    sequence_length: int = 30,
    prediction_days: int = 7,
    hidden_size: int = 256,
    num_layers: int = 3,
    station_embedding_dim: int = 8,
    season_embedding_dim: int = 4,
    num_epochs: int = 300,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    last_n_days: int = 1200,
    date_range_days_train: tuple = None,
    date_range_days_test: tuple = None,
    dropout: float = 0.2,
    exp_name=None,
    stride=1,
):
    """
    Train multi-station model on specified training stations
    
    Args:
        station_files: List of CSV file paths
        station_names: List of station names
        train_stations: List of station names to use for training
        test_station_name: Name of station to hold out for testing (optional)
        normalize_per_station: Whether to normalize each station separately
        existing_model: Pre-trained model to load (optional)
        use_seasons: Whether to use seasonal embeddings
    """
    
    ## Reload modules in case of changes
    reload(Models)
    reload(Trainer)
    reload(Plotter)
    reload(Analyzers)
    
    # Generate model filename based on training stations
    # Sort to ensure consistent naming regardless of order
    sorted_train_stations = sorted(train_stations)
    models_dir = f"{absolute_base_path}/results/models/"
    model_filename = (
        f"{models_dir}{exp_name}--{'_'.join(sorted_train_stations)}.pth"
    )
    
    print(f"Model filename: {model_filename}")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(model_filename) and existing_model is None:
        print(f"\nFound existing model: {model_filename}")
        print("Loading pre-trained model...")
        existing_model = torch.load(model_filename, weights_only=False)
        
        # Verify the model was trained on the same stations
        saved_train_stations = existing_model.get('training_stations', [])
        if set(saved_train_stations) == set(train_stations):
            print(f"Model was trained on stations: {', '.join(saved_train_stations)}")
        else:
            print(f"WARNING: Model was trained on different stations: {', '.join(saved_train_stations)}")
            print(f"Requested training stations: {', '.join(train_stations)}")
            print(f"Training new {model_type} model...")
            existing_model = None
    
    # Load all station data
    print("\nLoading station data...")
    all_station_data = Trainer.load_station_data(station_files, station_names)
    
    # Filter to only include training stations
    station_data = {name: data for name, data in all_station_data.items() if name in train_stations}
    
    # Validate that all requested training stations were found
    missing_stations = set(train_stations) - set(station_data.keys())
    if missing_stations:
        raise ValueError(f"Training stations not found in data: {missing_stations}")
    
    print(f"\nUsing {len(station_data)} stations for training: {', '.join(station_data.keys())}")
    
    # Handle test station
    if test_station_name and test_station_name in all_station_data:
        print(f"Holding out station '{test_station_name}' for testing")
        test_station_data = all_station_data[test_station_name]
                
        # Ensure test station is not in training data
        if test_station_name in station_data:
            # station_data.pop(test_station_name)
            # print(f"Removed {test_station_name} from training data")
            print(f"WARNING: Test station {test_station_name} is in the training data")
    else:
        test_station_data = None
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Seasonal embeddings: {'Enabled' if use_seasons else 'Disabled'}")
    
    ######################################
    # ============ Prepare data ==========
    #####################################
    print("\nPreparing multi-station data...")
    
    data_results = Trainer.prepare_multi_station_data(
        station_data, features, sequence_length, prediction_days, 
        normalize_per_station, last_n_days, date_range_days=date_range_days_train, use_seasons=use_seasons
    )   

    # Unpack results based on whether seasons are used
    if use_seasons:
        sequences, targets, station_ids, season_ids, station_to_id, scalers = data_results
    else:
        sequences, targets, station_ids, station_to_id, scalers = data_results
        season_ids = None
    
    print(f"\nTotal sequences: {len(sequences)}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Number of training stations: {len(station_data)}")
    
    # Print station distribution
    unique_stations, counts = np.unique(station_ids, return_counts=True)
    print("\nSequences per station:")
    for station_id, count in zip(unique_stations, counts):
        station_name = [k for k, v in station_to_id.items() if v == station_id][0]
        print(f"  {station_name}: {count} sequences")
    
    # Split data
    train_size = int(0.9 * len(sequences))
    val_size = int(0.1 * len(sequences))
    
    # Shuffle indices
    indices = np.random.permutation(len(sequences))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    
    # Create datasets
    if use_seasons:
        train_dataset = Models.MultiStationDataset(
            sequences[train_indices], 
            targets[train_indices], 
            station_ids[train_indices],
            season_ids[train_indices]
        )
        val_dataset = Models.MultiStationDataset(
            sequences[val_indices], 
            targets[val_indices], 
            station_ids[val_indices],
            season_ids[val_indices]
        )
    else:
        train_dataset = Models.MultiStationDataset(
            sequences[train_indices], 
            targets[train_indices], 
            station_ids[train_indices]
        )
        val_dataset = Models.MultiStationDataset(
            sequences[val_indices], 
            targets[val_indices], 
            station_ids[val_indices]
        )
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True,persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True,persistent_workers=True)
    
    # Initialize or load model
    num_stations = len(station_to_id)
    input_size = sequences.shape[2]
    output_size = prediction_days
    
    if existing_model is not None:
        # Extract the model configuration
        model_type = existing_model['type']
        model_config = existing_model['model_config']
        input_size = model_config['input_size']
        hidden_size = model_config['hidden_size']
        num_layers = model_config['num_layers']
        output_size = model_config['output_size']
        num_stations = model_config['num_stations']
        station_embedding_dim = model_config['station_embedding_dim']
        dropout = model_config.get('dropout', 0.5)
        use_seasons = model_config.get('use_seasons', False)
        
        # Check if the saved model has season support
        saved_use_seasons = model_config.get('use_seasons', False)
        saved_season_embedding_dim = model_config.get('season_embedding_dim', 4)
        
        if saved_use_seasons != use_seasons:
            print(f"\nWARNING: Model was trained with use_seasons={saved_use_seasons}, "
                  f"but current setting is use_seasons={use_seasons}")
            print("Using model's original season configuration...")
            use_seasons = saved_use_seasons
            season_embedding_dim = saved_season_embedding_dim

        if "lstm" in model_type:
            model = Models.MultiStationLSTM(
                input_size, hidden_size, num_layers, output_size, 
                num_stations, station_embedding_dim, dropout=dropout, 
                use_seasons=use_seasons, season_embedding_dim=season_embedding_dim
            ).to(device)  
        else:
            model = Models_Alt.get_model(model_type, input_size, hidden_size, num_layers, output_size, 
                num_stations, station_embedding_dim, dropout=dropout, 
                use_seasons=use_seasons, season_embedding_dim=season_embedding_dim
            ).to(device)

        # Load the model state dict
        model.load_state_dict(existing_model['model_state_dict'])
        
        # Load scalers and station mappings
        scalers = existing_model.get('scalers', scalers)
        station_to_id = existing_model.get('station_to_id', station_to_id)
        
        model.eval()
        print("Model loaded successfully")
        
        # Skip training
        print("\nSkipping training - using pre-trained model")
        train_losses, val_losses = None, None
    
    else: 

        
        
        
        
        ####################################
        # Train model
        ####################################
        print("\nTraining new model...")
        if 'lstm' in model_type:
            
            # config = Models_Alt.LSTMConfig(input_size=input_size,hidden_size=hidden_size, num_layers=num_layers,output_size=output_size,num_stations=num_stations,station_embedding_dim=station_embedding_dim,dropout=dropout,use_seasons=use_seasons,season_embedding_dim=season_embedding_dim, learning_rate=learning_rate)
            
            # model = Models_Alt.create_lstm_model(config, device)
            # print(
            #     f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters"
            #     )
            # print(f"Number of stations: {num_stations}")
            # print(f"Station Embedding dimension: {station_embedding_dim}")
            # if use_seasons:
            #     print(f"Season Embedding dimension: {season_embedding_dim}")
            
            # model, train_losses, val_losses = Trainer.train_lstm_optimized(
            #     model,
            #     train_loader,
            #     val_loader,
            #     config,
            #     device,
            #     num_epochs,
            #     patience
            # )    
            # 1. Try the simplest LSTM first
            # model = Models_Alt.SimplestLSTM(
            #     input_size=input_size,
            #     hidden_size=32,  # VERY SMALL
            #     num_layers=1,     # SINGLE LAYER
            #     output_size=output_size,
            #     num_stations=num_stations,
            #     station_embedding_dim=4,  # Smaller embeddings too
            #     use_seasons=use_seasons,
            #     dropout=0  # NO DROPOUT
            # ).to(device)
            
            model = Models.MultiStationLSTM(
                input_size, hidden_size, num_layers, output_size, 
                num_stations, station_embedding_dim, dropout=dropout,
                use_seasons=use_seasons, season_embedding_dim=season_embedding_dim
                ).to(device)
                
            print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters")
            print(f"Number of stations: {num_stations}")
            print(f"Station Embedding dimension: {station_embedding_dim}")
            if use_seasons:
                print(f"Season Embedding dimension: {season_embedding_dim}")
            
            ####################################
            # Train model
            ####################################
            print(f"\nTraining new {model_type}_{exp_name} model...")
            train_losses, val_losses = Trainer.train_multi_station_model(
                model, train_loader, val_loader, num_epochs, learning_rate, device, patience,
                use_seasons=use_seasons
            )
        else:
            model = Models_Alt.get_model(
                model_type,
                input_size,
                hidden_size,
                num_layers,
                output_size,
                num_stations,
                station_embedding_dim,
                dropout=dropout,
                use_seasons=use_seasons,
                season_embedding_dim=season_embedding_dim,
            ).to(device)
                        
            print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters")
            print(f"Number of stations: {num_stations}")
            print(f"Station Embedding dimension: {station_embedding_dim}")
            print(f"\nTraining new {model_type}_{exp_name} model...")
            if use_seasons:
                print(f"Season Embedding dimension: {season_embedding_dim}")
                train_losses, val_losses = Trainer.train_multi_station_model(
                    model, train_loader, val_loader, num_epochs, learning_rate, device, patience,
                    use_seasons=use_seasons
                )
            else:
                train_losses, val_losses = Trainer.train_multi_station_model(
                    model,
                    train_loader,
                    val_loader,
                    num_epochs,
                    learning_rate,
                    device,
                    patience
                )
    
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Multi-Station Model Training History')
        plt.legend()
        plt.grid(True)

        output_dir = "predictions"
        os.makedirs(output_dir, exist_ok=True)

        # Generate safe filename
        filename = f"model_training_history_{'_'.join(sorted_train_stations)}.png".replace(' ', '_')
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, bbox_inches='tight', dpi=200)
        print(f"Plot saved to: {filepath}")
        
        # Save model after training
        save_dict = {
            'type': model_type,
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'output_size': output_size,
                'num_stations': num_stations,
                'station_embedding_dim': station_embedding_dim,
                'use_seasons': use_seasons,
                'season_embedding_dim': season_embedding_dim if use_seasons else None,
                'dropout': dropout
            },
            'station_to_id': station_to_id,
            'scalers': scalers,
            'training_stations': sorted_train_stations,
            'normalize_per_station': normalize_per_station,
            'sequence_length': sequence_length,
            'prediction_days': prediction_days,
            'features': features,  # Save feature configuration
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        torch.save(save_dict, model_filename)
        print(f"\nModel saved to '{model_filename}'")
    
    # Analyze station embeddings
    # print("\nAnalyzing station embeddings...")
    # Analyzers.analyze_station_contributions(model, station_to_id, device)
    
    # # Feature importance analysis
    # Analyzers.debug_feature_importance_calculation(model, val_loader, features, device, use_seasons)
    # Analyzers.add_to_your_main_function(model, val_loader, features, device, use_seasons)

    ##########################
    # Test on held-out station
    ##########################
    mean_metrics = None
    if test_station_data is not None:
        print(f"\nTesting on held-out station: {test_station_name}")
        # Analyzers.comprehensive_correlation_analysis(test_station_data, features)

        # test_results = Trainer.test_on_new_station(
        #     model, features, test_station_data, test_station_name, station_to_id, 
        #     scalers, sequence_length, prediction_days, device, normalize_per_station, 
        #     last_n_days, date_range_days=date_range_days_test, use_seasons=use_seasons
        # )
        # metrics_path = f"resulsty/{exp_name}{test_station_name}/metrics_7_day"
        ancorhings = [None, 'replace', 'adjust', 'blend']
        
        for anchoring in ancorhings:
            metrics_path = f"resulsty/{exp_name}{test_station_name}/{anchoring}"
            
            if anchoring is None:
                test_results = Trainer.test_on_full_timeline(
                    model,
                    features,
                    test_station_data,
                    test_station_name,
                    station_to_id,
                    scalers,
                    sequence_length,
                    prediction_days,
                    device,
                    normalize_per_station,
                    stride=1,
                    use_seasons=use_seasons,
                    save_metrics_path=metrics_path,
                    anchor_predictions=False
                )
            else:
                test_results = Trainer.test_on_full_timeline(
                    model,
                    features,
                    test_station_data,
                    test_station_name,
                    station_to_id,
                    scalers,
                    sequence_length,
                    prediction_days,
                    device,
                    normalize_per_station,
                    stride=1,
                    use_seasons=use_seasons,
                    save_metrics_path=metrics_path,
                    anchor_predictions=True,
                    anchoring_method=anchoring,
                )
            
            # Visualize the results
            Trainer.visualize_timeline_predictions(
                test_results,
                sample_period_days=365,
                save_path=f"{metrics_path}/{exp_name}{test_station_name}_{anchoring}",
            )
            
            # Trainer.visualize_continuous_7day_forecasts(
            #     test_results,
            #     10,
            #     save_path=f"{metrics_path}/contiunous{exp_name}{test_station_name}_{anchoring}",
            # )
            
            # Trainer.visualize_forecast_comparison(
            #     test_results,
            #     save_path=f"{metrics_path}/companion{exp_name}{test_station_name}_{anchoring}",
            # )
            
            Trainer.visualize_forecast_sequence(
                test_results,
                start_date='2022-06-01',
                num_predictions=7,
                context_days_before=14,
                context_days_after=14,
                save_path=f"{metrics_path}/forecast_sequence{exp_name}{test_station_name}_{anchoring}",
            )
            
            Trainer.plot_full_timeline_simple(test_results,save_path=f"{metrics_path}/simple_full{exp_name}{test_station_name}_{anchoring}")
            
            Trainer.plot_timeline_with_residuals(
                test_results,
                save_path=f"{metrics_path}/residuals{exp_name}{test_station_name}_{anchoring}",
            )
            
            Trainer.plot_timeline_window(
                test_results,
                start_date="2022-01-01",
                window_days=365,
                show_metrics=True,
                save_path=f"{metrics_path}/window{exp_name}{test_station_name}_{anchoring}",
            )
            
            uncertainty_results = Trainer.analyze_prediction_uncertainty(
                test_results,
                save_path=f"{metrics_path}/{exp_name}{test_station_name}_{anchoring}_uncertainty_analysis",
            )

            # Create fan chart for a specific period
            Trainer.plot_prediction_fan_chart(
                test_results,
                start_date="2022-07-01",
                window_days=30,
                confidence_levels=[90, 75, 50],
                save_path=f"{metrics_path}/{exp_name}{test_station_name}_{anchoring}_fan_chart_july.png",
            )

            # Create comprehensive uncertainty plots
            Trainer.plot_uncertainty_analysis(
                uncertainty_results, save_path=f"{metrics_path}/{exp_name}{test_station_name}_{anchoring}_uncertainty"
            )

            # Access individual metrics
            print(f"Day 1 MAE: {test_results['metrics']['MAE']['day_1']:.4f}")
            print(f"Day 7 MAE: {test_results['metrics']['MAE']['day_7']:.4f}")
            print(f"Overall RMSE: {test_results['metrics']['MAE_overall']:.4f}")
            print(f"Overall RMSE: {test_results['metrics']['RMSE_overall']:.4f}")
            print(f"Overall RMSE: {test_results['metrics']['MAPE_overall']:.4f}")
            print(f"Overall R2: {test_results['metrics']['R2_overall']:.4f}")
            
    
    return model, scalers, station_to_id, mean_metrics, model_filename

# Add this function to detect available features from your data files
def detect_available_features(station_files, sample_size=5):
    """
    Detect which features are available across all station files
    """   
    all_features = set()
    common_features = None
    
    print("Detecting available features across all stations...")
    
    # Sample a few files to check features
    files_to_check = station_files[:sample_size] if len(station_files) > sample_size else station_files
    
    for i, file_path in enumerate(files_to_check):
        try:
            df = pd.read_csv(file_path, nrows=10)  # Just read header and few rows
            features = set(df.columns)
            all_features.update(features)
            
            if common_features is None:
                common_features = features
            else:
                common_features = common_features.intersection(features)
            
            print(f"\nFile {i+1}: {os.path.basename(file_path)}")
            print(f"Features: {sorted(features)}")
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"\n{'='*60}")
    print("FEATURE SUMMARY:")
    print(f"{'='*60}")
    print(f"Total unique features found: {len(all_features)}")
    print(f"Common features in all files: {len(common_features)}")
    print(f"\nCommon features: {sorted(common_features)}")
    
    return sorted(common_features), sorted(all_features)

def run_single_experiment(
    model_name, exp, station_files, water_stations, train_stations, test_station, stride=1
):
    """Run single experiment"""
    
    # test_station = test_stations[0]
    
    
    try:
        # Run the experiment with all parameters
        _, _, _, mean_metrics, model_filename = main(
            model_name,
            station_files, 
            water_stations, 
            exp['features'],
            train_stations = train_stations,
            test_station_name=test_station,
            normalize_per_station=exp['normalize_per_station'],
            existing_model=None,
            use_seasons=exp['use_seasons'],
            use_old=exp['use_old'],
            patience=exp['patience'],
            sequence_length=exp['sequence_length'],
            prediction_days=exp['prediction_days'],
            hidden_size=exp['hidden_size'],
            num_layers=exp['num_layers'],
            station_embedding_dim=exp['station_embedding_dim'],
            season_embedding_dim=exp['season_embedding_dim'],
            num_epochs=exp['num_epochs'],
            batch_size=exp['batch_size'],
            learning_rate=exp['learning_rate'],
            last_n_days=exp['last_n_days'],
            date_range_days_train=exp['date_range_train'],
            date_range_days_test=exp['date_range_test'],
            dropout=exp['dropout'],
            exp_name=f"{exp['name']}",
            stride = stride
        )
        
        if 'torch' in sys.modules:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # If using matplotlib
        if 'matplotlib.pyplot' in sys.modules:
            plt.close('all')

        # Force garbage collection
        gc.collect()
        
        
    except Exception as e:
        print(f"❌ Failed: {exp['name']} - {test_station}")
        print(f"   Error: {str(e)}")
                


# MAIN EXPERIMENT RUNNER WITH RESUME CAPABILITY
def run_experiments_with_resume(experiments, station_files, water_stations, train_stations, test_stations):
    """Run experiments with stop/resume capability"""
    
    # Load progress
    progress = load_progress()
    completed_ids = set(progress['completed_experiments'])
    
    # Update total experiments if not set
    if progress['total_experiments'] == 0:
        total_combinations = len(experiments) * len(test_stations)
        progress['total_experiments'] = total_combinations
        save_progress(progress)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT RUNNER")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Test stations: {len(test_stations)}")
    print(f"Total combinations: {progress['total_experiments']}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Remaining: {progress['total_experiments'] - len(completed_ids)}")
    print(f"{'='*80}")
    
    if len(completed_ids) > 0:
        print("\n📌 Resuming from previous run...")
        print(f"Last completed: {progress['last_completed']}")
    
    print("\n💡 To stop gracefully:")
    print("   - Press Ctrl+C once")
    print("   - Or create a file named 'STOP_EXPERIMENTS.txt' in the current directory")
    print(f"\n{'='*80}\n")
    
    # Remove stop file if it exists from previous run
    if os.path.exists('STOP_EXPERIMENTS.txt'):
        os.remove('STOP_EXPERIMENTS.txt')
    
    test_stations_to_run = test_stations
    experiments_run = 0
    experiments_skipped = 0
    
    # Run all experiments
    for exp_idx, exp in enumerate(experiments):
        experiment_id = exp_idx + 1
        
        # Check if we should stop
        if stop_requested or check_stop_file():
            print(f"\n{'='*80}")
            print("🛑 STOPPING EXPERIMENTS")
            print(f"{'='*80}")
            print(f"Experiments completed in this run: {experiments_run}")
            print(f"Experiments skipped (already done): {experiments_skipped}")
            print(f"Total completed overall: {len(completed_ids)}")
            print(f"Remaining: {progress['total_experiments'] - len(completed_ids)}")
            print("\nTo resume, run the script again.")
            break
        
        # Check if all stations for this experiment are done
        all_stations_done = True
        for test_station in test_stations_to_run:
            exp_unique_id = get_experiment_id(experiment_id, exp['name'], test_station)
            if exp_unique_id not in completed_ids:
                all_stations_done = False
                break
        
        if all_stations_done:
            experiments_skipped += len(test_stations_to_run)
            continue
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {experiment_id}/{len(experiments)}: {exp['name']}")
        print(f"{'='*80}")
        
        for test_station_n, test_station in enumerate(test_stations_to_run):
            # Check stop conditions again
            if stop_requested or check_stop_file():
                break
            
            # Generate unique ID for this experiment-station combination
            exp_unique_id = get_experiment_id(experiment_id, exp['name'], test_station)
            
            # Skip if already completed
            if exp_unique_id in completed_ids:
                print(f"⏩ Skipping (already done): {exp['name']} - {test_station}")
                experiments_skipped += 1
                continue
            
            print(f"\n🔄 Running: {exp['name']} - {test_station}")
            print(f"   Progress: {len(completed_ids) + 1}/{progress['total_experiments']}")
            
            try:
                # Save experiment info
                exp_dir = save_experiment_info(experiment_id, exp['name'], exp, test_station, test_station_n + 1)
                
                # Create subdirectories for this experiment
                os.makedirs(f"{exp_dir}/models", exist_ok=True)
                os.makedirs(f"{exp_dir}/predictions", exist_ok=True)
                
                # Temporarily change working directory
                original_dir = os.getcwd()
                os.chdir(exp_dir)
                
                try:
                    # Run the experiment with all parameters
                    _, _, _, mean_metrics, model_filename = main(
                        station_files, 
                        water_stations, 
                        exp['features'],
                        train_stations = train_stations,
                        test_station_name=test_station,
                        normalize_per_station=exp['normalize_per_station'],
                        existing_model=None,
                        use_seasons=exp['use_seasons'],
                        use_old=exp['use_old'],
                        patience=exp['patience'],
                        sequence_length=exp['sequence_length'],
                        prediction_days=exp['prediction_days'],
                        hidden_size=exp['hidden_size'],
                        num_layers=exp['num_layers'],
                        station_embedding_dim=exp['station_embedding_dim'],
                        season_embedding_dim=exp['season_embedding_dim'],
                        num_epochs=exp['num_epochs'],
                        batch_size=exp['batch_size'],
                        learning_rate=exp['learning_rate'],
                        last_n_days=exp['last_n_days'],
                        date_range_days_train=exp['date_range_train'],
                        date_range_days_test=exp['date_range_test'],
                        dropout=exp['dropout'],
                        exp_name=f"{exp['name']}"
                    )
                    
                    os.chdir(original_dir)
                    # Mark as completed
                    completed_ids.add(exp_unique_id)
                    progress['completed_experiments'] = list(completed_ids)
                    progress['last_completed'] = exp_unique_id
                    progress['last_update'] = str(datetime.now())
                    
                    save_progress(progress)
                    
                    Metrics.save_metrics_append(f"{original_dir}/results/metrics", experiment_id, exp['name'], test_station, test_station_n , mean_metrics, model_filename)
                    
                    del mean_metrics
                    del model_filename
                    
                    if 'torch' in sys.modules:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    # If using matplotlib
                    if 'matplotlib.pyplot' in sys.modules:
                        plt.close('all')

                    # Force garbage collection
                    gc.collect()
                    
                    experiments_run += 1
                    print(f"✅ Completed: {exp['name']} - {test_station}")
                    
                except Exception as e:
                    print(f"❌ Failed: {exp['name']} - {test_station}")
                    print(f"   Error: {str(e)}")
                    
                    # Save failure info
                    os.chdir(original_dir)
                    failure_file = "results/experiments/failed_experiments.txt"
                    with open(failure_file, 'a') as f:
                        f.write(f"\n{'='*50}\n")
                        f.write(f"Failed at: {datetime.now()}\n")
                        f.write(f"Experiment: {exp['name']}\n")
                        f.write(f"Experiment ID: {experiment_id}\n")
                        f.write(f"Test Station: {test_station}\n")
                        f.write(f"Error: {str(e)}\n")
                        f.write(f"Unique ID: {exp_unique_id}\n")
                        import traceback
                        f.write(f"Traceback:\n{traceback.format_exc()}\n")
                    
                    # Don't mark as completed so it can be retried
                    continue
                
                finally:
                    # Always change back to original directory
                    os.chdir(original_dir)
                
            except Exception as e:
                print(f"❌ Critical error in experiment setup: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Ensure we're back in original directory
                if os.getcwd() != original_dir:
                    os.chdir(original_dir)
    
    # Save final summary
    summary = {
        'total_experiments': len(experiments),
        'test_stations': test_stations_to_run,
        'completed_at': str(datetime.now()),
        'experiments_completed': len(completed_ids),
        'experiments_failed': progress['total_experiments'] - len(completed_ids),
        'experiments': [exp['name'] for exp in experiments],
        'completed_ids': list(completed_ids)
    }
    
    with open('results/experiments/final_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT RUN COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {datetime.now()}")
    print(f"Experiments run in this session: {experiments_run}")
    print(f"Experiments skipped (already done): {experiments_skipped}")
    print(f"Total completed overall: {len(completed_ids)}")
    print(f"Total remaining: {progress['total_experiments'] - len(completed_ids)}")
    print("Results saved in 'experiments' directory")
    print(f"{'='*80}")

def check_experiment_progress():
    """Check current progress without running experiments"""
    progress = load_progress()
    completed = len(progress['completed_experiments'])
    total = progress['total_experiments']
    
    print(f"\n{'='*80}")
    print("EXPERIMENT PROGRESS")
    print(f"{'='*80}")
    print(f"Total experiments: {total}")
    print(f"Completed: {completed}")
    print(f"Remaining: {total - completed}")
    print(f"Progress: {(completed/total*100):.1f}%" if total > 0 else "No experiments configured")
    
    if progress.get('last_completed'):
        print(f"\nLast completed: {progress['last_completed']}")
        print(f"Last update: {progress.get('last_update', 'Unknown')}")
    
    if completed > 0:
        print("\nCompleted experiments:")
        for i, exp_id in enumerate(sorted(progress['completed_experiments'])[:10]):
            print(f"  {i+1}. {exp_id}")
        if completed > 10:
            print(f"  ... and {completed - 10} more")
    
    print(f"{'='*80}\n")


feature_sets = {
    "minimal": ["water_level", "API"],
    "add_lag": ["water_level", "water_level_lag_1d", "API"],
    "add_temp": ["water_level", "API", "Temp_4day_lagged"],
    "add_sun": ["water_level", "API", "Sun"],
    "add_pressure": ["water_level", "API", "Temp_4day_lagged", "Sun", "Pressure"],
    "standard": [
        "water_level",
        "water_level_lag_1d",
        "API",
        "rainfall_sum_7d",
        "Temp",
        "Temp_4day_lagged",
    ],
    "rainfall_focus": [
        "water_level_lag_1d",
        "Rainfall_day",
        "rainfall_sum_3d",
        "rainfall_sum_7d",
        "rainfall_sum_14d",
        "rainfall_sum_30d",
        "API",
    ],
    "weather_basic": [
        "water_level",
        "water_level_lag_1d",
        "rainfall_sum_14d",
        "Temp",
        "Pressure",
        "Wind_speed",
        "Sun",
    ],
    "weather_full": [
        "water_level",
        "water_level_lag_1d",
        "rainfall_sum_14d",
        "API",
        "Temp",
        "Temp_4day_lagged",
        "Pressure",
        "pressure_change_3d",
        "Wind_speed",
        "Sun",
        "Humidity",
    ],
}


experiments = [
    # {
    #     "model_name": "lstm",
    #     "model_type": "lstm",
    #     "name": "LSTM",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 40,
    #     "prediction_days": 7,
    #     "hidden_size": 128,
    #     "num_layers": 2,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 32,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.3,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "lstm",
    #     "model_type": "lstm",
    #     "name": "lstm256_3",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 40,
    #     "prediction_days": 7,
    #     "hidden_size": 256,
    #     "num_layers": 3,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 64,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.3,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "lstm",
    #     "model_type": "lstm",
    #     "name": "lstm512_2_seq40",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 40,
    #     "prediction_days": 7,
    #     "hidden_size": 512,
    #     "num_layers": 2,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 64,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.3,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "lstm",
    #     "model_type": "lstm",
    #     "name": "lstm512_2_seq20",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 20,
    #     "prediction_days": 7,
    #     "hidden_size": 512,
    #     "num_layers": 2,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 64,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.3,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "lstm",
    #     "model_type": "lstm",
    #     "name": "lstm512_2_seq30",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 30,
    #     "prediction_days": 7,
    #     "hidden_size": 512,
    #     "num_layers": 2,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 64,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.3,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "lstm",
    #     "model_type": "lstm",
    #     "name": "lstm512_2_seq60",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 60,
    #     "prediction_days": 7,
    #     "hidden_size": 512,
    #     "num_layers": 2,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 64,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.3,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "lstm",
    #     "model_type": "lstm",
    #     "name": "lstm512_2_pred_3",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 40,
    #     "prediction_days": 3,
    #     "hidden_size": 512,
    #     "num_layers": 2,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 128,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.3,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "lstm",
    #     "model_type": "lstm",
    #     "name": "lstm512_2_pred_3",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 60,
    #     "prediction_days": 14,
    #     "hidden_size": 256,
    #     "num_layers": 2,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 64,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.3,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "lstm_seasons",
    #     "model_type": "lstm",
    #     "name": "LSTM_Seasons",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 40,
    #     "prediction_days": 7,
    #     "hidden_size": 128,
    #     "num_layers": 2,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 64,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.3,
    #     "patience": 50,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "transformer",
    #     "model_type": "transformer",
    #     "name": "transformer",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 40,
    #     "prediction_days": 7,
    #     "hidden_size": 256,
    #     "num_layers": 3,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 128,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.1,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "transformer",
    #     "model_type": "transformer",
    #     "name": "transformer_h512_L3",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 40,
    #     "prediction_days": 7,
    #     "hidden_size": 512,
    #     "num_layers": 3,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 128,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.1,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "transformer",
    #     "model_type": "transformer",
    #     "name": "transformer_seq60_h512_L3",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 60,
    #     "prediction_days": 7,
    #     "hidden_size": 512,
    #     "num_layers": 3,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 128,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.1,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    {
        "model_name": "transformer",
        "model_type": "transformer",
        "name": "transformer_minimal",
        "features": feature_sets["minimal"],
        "sequence_length": 40,
        "prediction_days": 7,
        "hidden_size": 512,
        "num_layers": 2,
        "station_embedding_dim": 8,
        "season_embedding_dim": 4,
        "num_epochs": 300,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "dropout": 0.1,
        "patience": 20,
        "normalize_per_station": True,
        "use_seasons": False,
        "use_old": False,
        "date_range_train": None,
        "last_n_days": 0,
        "date_range_test": None,
    },
    {
        "model_name": "transformer",
        "model_type": "transformer",
        "name": "transformer_add_lag",
        "features": feature_sets["add_lag"],
        "sequence_length": 40,
        "prediction_days": 7,
        "hidden_size": 512,
        "num_layers": 2,
        "station_embedding_dim": 8,
        "season_embedding_dim": 4,
        "num_epochs": 300,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "dropout": 0.1,
        "patience": 20,
        "normalize_per_station": True,
        "use_seasons": False,
        "use_old": False,
        "date_range_train": None,
        "last_n_days": 0,
        "date_range_test": None,
    },
    {
        "model_name": "transformer",
        "model_type": "transformer",
        "name": "transformer_add_temp",
        "features": feature_sets["add_temp"],
        "sequence_length": 40,
        "prediction_days": 7,
        "hidden_size": 512,
        "num_layers": 2,
        "station_embedding_dim": 8,
        "season_embedding_dim": 4,
        "num_epochs": 300,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "dropout": 0.1,
        "patience": 20,
        "normalize_per_station": True,
        "use_seasons": False,
        "use_old": False,
        "date_range_train": None,
        "last_n_days": 0,
        "date_range_test": None,
    },
    {
        "model_name": "transformer",
        "model_type": "transformer",
        "name": "transformer_add_sun",
        "features": feature_sets["add_sun"],
        "sequence_length": 40,
        "prediction_days": 7,
        "hidden_size": 512,
        "num_layers": 2,
        "station_embedding_dim": 8,
        "season_embedding_dim": 4,
        "num_epochs": 300,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "dropout": 0.1,
        "patience": 20,
        "normalize_per_station": True,
        "use_seasons": False,
        "use_old": False,
        "date_range_train": None,
        "last_n_days": 0,
        "date_range_test": None,
    },
    {
        "model_name": "transformer",
        "model_type": "transformer",
        "name": "transformer_add_pressure",
        "features": feature_sets["add_pressure"],
        "sequence_length": 40,
        "prediction_days": 7,
        "hidden_size": 512,
        "num_layers": 2,
        "station_embedding_dim": 8,
        "season_embedding_dim": 4,
        "num_epochs": 300,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "dropout": 0.1,
        "patience": 20,
        "normalize_per_station": True,
        "use_seasons": False,
        "use_old": False,
        "date_range_train": None,
        "last_n_days": 0,
        "date_range_test": None,
    },
    {
        "model_name": "transformer",
        "model_type": "transformer",
        "name": "transformer_standard",
        "features": feature_sets["standard"],
        "sequence_length": 40,
        "prediction_days": 7,
        "hidden_size": 512,
        "num_layers": 2,
        "station_embedding_dim": 8,
        "season_embedding_dim": 4,
        "num_epochs": 300,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "dropout": 0.1,
        "patience": 20,
        "normalize_per_station": True,
        "use_seasons": False,
        "use_old": False,
        "date_range_train": None,
        "last_n_days": 0,
        "date_range_test": None,
    },
    {
        "model_name": "transformer",
        "model_type": "transformer",
        "name": "transformer_rainfall_focus",
        "features": feature_sets["rainfall_focus"],
        "sequence_length": 40,
        "prediction_days": 7,
        "hidden_size": 512,
        "num_layers": 2,
        "station_embedding_dim": 8,
        "season_embedding_dim": 4,
        "num_epochs": 300,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "dropout": 0.1,
        "patience": 20,
        "normalize_per_station": True,
        "use_seasons": False,
        "use_old": False,
        "date_range_train": None,
        "last_n_days": 0,
        "date_range_test": None,
    },
    {
        "model_name": "transformer",
        "model_type": "transformer",
        "name": "transformer_weather_full",
        "features": feature_sets["weather_full"],
        "sequence_length": 40,
        "prediction_days": 7,
        "hidden_size": 512,
        "num_layers": 2,
        "station_embedding_dim": 8,
        "season_embedding_dim": 4,
        "num_epochs": 300,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "dropout": 0.1,
        "patience": 20,
        "normalize_per_station": True,
        "use_seasons": False,
        "use_old": False,
        "date_range_train": None,
        "last_n_days": 0,
        "date_range_test": None,
    },
    # {
    #     "model_name": "transformer",
    #     "model_type": "transformer",
    #     "name": "transformer_h512_L2_seq20",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 20,
    #     "prediction_days": 7,
    #     "hidden_size": 512,
    #     "num_layers": 2,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 128,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.1,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "transformer",
    #     "model_type": "transformer",
    #     "name": "transformer_h512_L2_seq30",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 30,
    #     "prediction_days": 7,
    #     "hidden_size": 512,
    #     "num_layers": 2,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 128,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.1,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "transformer",
    #     "model_type": "transformer",
    #     "name": "transformer_h512_L2_seq60",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 60,
    #     "prediction_days": 7,
    #     "hidden_size": 512,
    #     "num_layers": 2,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 128,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.1,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "transformer",
    #     "model_type": "transformer",
    #     "name": "transformer_pred14",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 40,
    #     "prediction_days": 14,
    #     "hidden_size": 256,
    #     "num_layers": 3,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 128,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.1,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "transformer",
    #     "model_type": "transformer",
    #     "name": "transformer_pred14_h512_L3",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 60,
    #     "prediction_days": 14,
    #     "hidden_size": 512,
    #     "num_layers": 3,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 128,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.1,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "transformer",
    #     "model_type": "transformer",
    #     "name": "transformer_seq_60pred14_h512_L2",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 60,
    #     "prediction_days": 14,
    #     "hidden_size": 512,
    #     "num_layers": 2,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 128,
    #     "learning_rate": 0.0001,
    #     "dropout": 0.1,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "gru",
    #     "model_type": "gru",
    #     "name": "gru",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 40,
    #     "prediction_days": 7,
    #     "hidden_size": 256,
    #     "num_layers": 3,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 128,
    #     "learning_rate": 0.001,
    #     "dropout": 0.3,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "cnn",
    #     "model_type": "cnn",
    #     "name": "cnn",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 40,
    #     "prediction_days": 7,
    #     "hidden_size": 128,
    #     "num_layers": 3,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 128,
    #     "learning_rate": 0.001,
    #     "dropout": 0.2,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
    # {
    #     "model_name": "ffnn",
    #     "model_type": "ffnn",
    #     "name": "ffnn",
    #     "features": feature_sets["standard"],
    #     "sequence_length": 40,
    #     "prediction_days": 7,
    #     "hidden_size": 512,
    #     "num_layers": 4,
    #     "station_embedding_dim": 8,
    #     "season_embedding_dim": 4,
    #     "num_epochs": 300,
    #     "batch_size": 128,
    #     "learning_rate": 0.001,
    #     "dropout": 0.5,
    #     "patience": 20,
    #     "normalize_per_station": True,
    #     "use_seasons": False,
    #     "use_old": False,
    #     "date_range_train": None,
    #     "last_n_days": 0,
    #     "date_range_test": None,
    # },
   # },
]



if __name__ == "__main__":
    # Check if we just want to see progress
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        check_experiment_progress()
    else:
        # Run experiments with resume capability
        
        STATION_MAPPINGS = load_mappings()
        print(STATION_MAPPINGS)

        import Experiment_Generator
        reload(Experiment_Generator)

        # experiments, base_experiments, date_ranges = Experiment_Generator.generate_all_experiments()

        # print(f"Total number of experiments: {len(experiments)}")
        # print(f"Base experiments: {len(base_experiments)}")
        # print(f"Date ranges: {len(date_ranges)}")
        # print(f"Total combinations: {len(base_experiments)} × {len(date_ranges)} = {len(experiments)}")


        # print(date_ranges)

        CONFIG = load_config()
        print(CONFIG['base_path_data'])
        base_path_data = CONFIG['base_path_data']

        base_path = f"{base_path_data}\\1_combined_weather_water_level\\"

        water_stations = []
        station_files = []

        for weather_station, water_stations_names in STATION_MAPPINGS.items():
            for water_station_name in water_stations_names:
                water_stations.append(water_station_name)
                station_files.append(f"{base_path}preprocessed_{weather_station}_{water_station_name}.csv")

        print(f"Files loaded: {station_files}")
        
        common_features, all_features = detect_available_features(station_files)
        
        
        station_files, water_stations_train, water_stations_test = Station_Train_Test_Setup.get_train_test_stations()
        
        
        # Filter experiments to only use available features
        # valid_experiments = []
        # for exp in experiments:
        #     # Check if all experiment features are available
        #     exp_features = set(exp['features'])
        #     if exp_features.issubset(set(common_features)):
        #         valid_experiments.append(exp)
        #     else:
        #         missing = exp_features - set(common_features)
        #         print(f"\nSkipping experiment '{exp['name']}' - missing features: {missing}")

        # print(f"\n{len(valid_experiments)} out of {len(experiments)} experiments are valid with available features")
        # experiments = valid_experiments
        
        # run_experiments_with_resume(experiments, station_files, water_stations, water_stations_train, water_stations_test)
        # test_station = "Sengelose_test"
        test_station = 'Vaerebro'
        # test_station = 'Hove'
        # test_station = 'Gudena'
        # test_station = 'Vorgod'
        models_to_test = ["lstm", "lstm2", "transformer", "gru", "cnn","ffnn"]
        testing_stations = ['Sengelose_test', 'Hove', 'Vaerebro', 'Gudena', 'Vorgod']
        for exp in experiments:
            if 'stride' in exp.keys():
                stride = exp['stride']
            else:
                stride = 1
            for test_station in testing_stations:
                model_type = exp["model_type"]
                run_single_experiment(
                    model_type,
                    exp,
                    station_files,
                    water_stations,
                    water_stations_train,
                    test_station,
                    stride=1,
                )

