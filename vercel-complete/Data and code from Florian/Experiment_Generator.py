date_ranges_train = [
    None,  # Use all available data
    (-365, 0),    # Last 300 days
    (-710, 0),    # Last 600 days
    (-1420, 0),    # Last 900 days
    (-2400, 0),   # Last 2400 days
]

date_ranges_test = [
    None,  # Use all available data
    (-1000, -100),    # Last 300 days
    (-1000, -200),    # Last 300 days
    (-1000, -300)
]

# Define parameter ranges
sequence_lengths = [14, 30, 40, 60, 90, 120]
prediction_days = [1, 2, 4, 6, 7, 10, 14]
hidden_sizes = [64, 128, 256, 512, 750, 1024]
num_layers = [1, 2, 3, 4, 5, 6, 7]
station_embedding_dims = [5, 8, 12]
num_epochs = [100, 150, 200, 250, 300, 400]
learning_rates = [0.0001, 0.001, 0.005, 0.00001]
dropouts = [0, 0.2, 0.4, 0.5, 0.7]
patiences = [10, 20, 30]

# Define feature sets
feature_sets = {
    'minimal': ['water_level_lag_1d', 'rainfall_sum_14d'],
    
    'basic': ['water_level', 'water_level_lag_1d', 'rainfall_sum_14d', 'Temp'],
    
    'standard': ['water_level', 'water_level_lag_1d', 'water_level_lag_3d', 
                 'rainfall_sum_14d', 'Temp', 'API'],
    
    'extended_water': ['water_level', 'water_level_lag_1d', 'water_level_lag_2d', 
                       'water_level_lag_3d', 'water_level_lag_7d', 'water_level_lag_14d',
                       'rainfall_sum_14d', 'API'],
    
    'rainfall_focus': ['water_level_lag_1d', 'Rainfall_day', 'rainfall_sum_3d', 
                       'rainfall_sum_7d', 'rainfall_sum_14d', 'rainfall_sum_30d', 'API'],
    
    'weather_basic': ['water_level_lag_1d', 'rainfall_sum_14d', 'Temp', 'Pressure', 
                      'Wind_speed', 'Sun'],
    
    'weather_full': ['water_level', 'water_level_lag_1d', 'rainfall_sum_14d', 'API',
                     'Temp', 'Temp_4day_lagged', 'Pressure', 'pressure_change_3d',
                     'Wind_speed', 'Sun', 'Humidity'],
    
    'temperature_focus': ['water_level_lag_1d', 'rainfall_sum_14d', 'Temp', 
                          'Temp_min_mean', 'Temp_max', 'Temp_soil_mean', 'Temp_4day_lagged'],
    
    'pressure_focus': ['water_level_lag_1d', 'rainfall_sum_14d', 'Pressure', 
                       'pressure_change_1d', 'pressure_change_3d', 'pressure_trend_5d'],
    
    'seasonal': ['water_level', 'water_level_lag_1d', 'rainfall_sum_14d', 
                 'Temp', 'API', 'Season'],
    
    'comprehensive': ['water_level', 'water_level_lag_1d', 'water_level_lag_3d', 
                      'water_level_lag_7d', 'rainfall_sum_14d', 'rainfall_sum_30d',
                      'API', 'Temp', 'Temp_4day_lagged', 'Pressure', 'pressure_change_3d',
                      'Wind_speed', 'Sun', 'Humidity', 'Season'],
}

# Create base experiments with different architectural approaches
base_experiments = [
    # TINY MODELS - Fast training, baseline performance
    {
        'name': 'tiny_minimal',
        'features': feature_sets['minimal'],
        'sequence_length': 14,
        'prediction_days': 7,
        'hidden_size': 64,
        'num_layers': 1,
        'station_embedding_dim': 5,
        'season_embedding_dim': 4,
        'num_epochs': 100,
        'batch_size': 512,
        'learning_rate': 0.005,
        'dropout': 0.2,
        'patience': 10,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    {
        'name': 'tiny_basic',
        'features': feature_sets['basic'],
        'sequence_length': 30,
        'prediction_days': 7,
        'hidden_size': 128,
        'num_layers': 2,
        'station_embedding_dim': 5,
        'season_embedding_dim': 4,
        'num_epochs': 150,
        'batch_size': 256,
        'learning_rate': 0.001,
        'dropout': 0.4,
        'patience': 20,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    
    # SMALL MODELS - Good balance of speed and performance
    {
        'name': 'small_standard',
        'features': feature_sets['standard'],
        'sequence_length': 30,
        'prediction_days': 7,
        'hidden_size': 128,
        'num_layers': 2,
        'station_embedding_dim': 8,
        'season_embedding_dim': 4,
        'num_epochs': 200,
        'batch_size': 256,
        'learning_rate': 0.001,
        'dropout': 0.2,
        'patience': 20,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    {
        'name': 'small_weather',
        'features': feature_sets['weather_basic'],
        'sequence_length': 40,
        'prediction_days': 7,
        'hidden_size': 256,
        'num_layers': 2,
        'station_embedding_dim': 8,
        'season_embedding_dim': 4,
        'num_epochs': 200,
        'batch_size': 256,
        'learning_rate': 0.001,
        'dropout': 0.2,
        'patience': 20,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    
    # MEDIUM MODELS - Standard configurations
    {
        'name': 'medium_standard',
        'features': feature_sets['standard'],
        'sequence_length': 40,
        'prediction_days': 7,
        'hidden_size': 256,
        'num_layers': 3,
        'station_embedding_dim': 8,
        'season_embedding_dim': 4,
        'num_epochs': 250,
        'batch_size': 128,
        'learning_rate': 0.001,
        'dropout': 0.2,
        'patience': 30,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    {
        'name': 'medium_rainfall',
        'features': feature_sets['rainfall_focus'],
        'sequence_length': 60,
        'prediction_days': 7,
        'hidden_size': 256,
        'num_layers': 3,
        'station_embedding_dim': 8,
        'season_embedding_dim': 4,
        'num_epochs': 300,
        'batch_size': 128,
        'learning_rate': 0.0001,
        'dropout': 0.4,
        'patience': 30,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    {
        'name': 'medium_seasonal',
        'features': feature_sets['seasonal'],
        'sequence_length': 40,
        'prediction_days': 7,
        'hidden_size': 256,
        'num_layers': 3,
        'station_embedding_dim': 8,
        'season_embedding_dim': 8,
        'num_epochs': 300,
        'batch_size': 128,
        'learning_rate': 0.001,
        'dropout': 0.2,
        'patience': 30,
        'normalize_per_station': True,
        'use_seasons': True,
        'use_old': False,
    },
    
    # LARGE MODELS - More complex architectures
    {
        'name': 'large_standard',
        'features': feature_sets['standard'],
        'sequence_length': 60,
        'prediction_days': 7,
        'hidden_size': 512,
        'num_layers': 4,
        'station_embedding_dim': 12,
        'season_embedding_dim': 4,
        'num_epochs': 300,
        'batch_size': 64,
        'learning_rate': 0.0001,
        'dropout': 0.2,
        'patience': 30,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    {
        'name': 'large_weather',
        'features': feature_sets['weather_full'],
        'sequence_length': 90,
        'prediction_days': 7,
        'hidden_size': 512,
        'num_layers': 4,
        'station_embedding_dim': 12,
        'season_embedding_dim': 4,
        'num_epochs': 400,
        'batch_size': 64,
        'learning_rate': 0.0001,
        'dropout': 0.4,
        'patience': 30,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    {
        'name': 'large_comprehensive',
        'features': feature_sets['comprehensive'],
        'sequence_length': 90,
        'prediction_days': 7,
        'hidden_size': 750,
        'num_layers': 5,
        'station_embedding_dim': 12,
        'season_embedding_dim': 8,
        'num_epochs': 400,
        'batch_size': 32,
        'learning_rate': 0.00001,
        'dropout': 0.4,
        'patience': 30,
        'normalize_per_station': True,
        'use_seasons': True,
        'use_old': False,
    },
    
    # EXTRA LARGE MODELS
    {
        'name': 'xlarge_deep',
        'features': feature_sets['comprehensive'],
        'sequence_length': 120,
        'prediction_days': 7,
        'hidden_size': 1024,
        'num_layers': 6,
        'station_embedding_dim': 12,
        'season_embedding_dim': 8,
        'num_epochs': 400,
        'batch_size': 32,
        'learning_rate': 0.00001,
        'dropout': 0.5,
        'patience': 30,
        'normalize_per_station': True,
        'use_seasons': True,
        'use_old': False,
    },
    
    # SPECIALIZED EXPERIMENTS
    
    # Short-term prediction (1-2 days)
    {
        'name': 'short_term_1day',
        'features': feature_sets['basic'],
        'sequence_length': 14,
        'prediction_days': 1,
        'hidden_size': 128,
        'num_layers': 2,
        'station_embedding_dim': 5,
        'season_embedding_dim': 4,
        'num_epochs': 150,
        'batch_size': 512,
        'learning_rate': 0.001,
        'dropout': 0.2,
        'patience': 20,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    {
        'name': 'short_term_2day',
        'features': feature_sets['standard'],
        'sequence_length': 30,
        'prediction_days': 2,
        'hidden_size': 256,
        'num_layers': 3,
        'station_embedding_dim': 8,
        'season_embedding_dim': 4,
        'num_epochs': 200,
        'batch_size': 256,
        'learning_rate': 0.001,
        'dropout': 0.2,
        'patience': 20,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    
    # Medium-term prediction (4-6 days)
    {
        'name': 'medium_term_4day',
        'features': feature_sets['standard'],
        'sequence_length': 40,
        'prediction_days': 4,
        'hidden_size': 256,
        'num_layers': 3,
        'station_embedding_dim': 8,
        'season_embedding_dim': 4,
        'num_epochs': 250,
        'batch_size': 128,
        'learning_rate': 0.0001,
        'dropout': 0.4,
        'patience': 30,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    {
        'name': 'medium_term_6day',
        'features': feature_sets['weather_basic'],
        'sequence_length': 60,
        'prediction_days': 6,
        'hidden_size': 512,
        'num_layers': 4,
        'station_embedding_dim': 12,
        'season_embedding_dim': 4,
        'num_epochs': 300,
        'batch_size': 64,
        'learning_rate': 0.0001,
        'dropout': 0.4,
        'patience': 30,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    
    # Long-term prediction (10-14 days)
    {
        'name': 'long_term_10day',
        'features': feature_sets['weather_full'],
        'sequence_length': 90,
        'prediction_days': 10,
        'hidden_size': 512,
        'num_layers': 5,
        'station_embedding_dim': 12,
        'season_embedding_dim': 8,
        'num_epochs': 300,
        'batch_size': 64,
        'learning_rate': 0.0001,
        'dropout': 0.5,
        'patience': 30,
        'normalize_per_station': True,
        'use_seasons': True,
        'use_old': False,
    },
    {
        'name': 'long_term_14day',
        'features': feature_sets['comprehensive'],
        'sequence_length': 120,
        'prediction_days': 14,
        'hidden_size': 512,
        'num_layers': 6,
        'station_embedding_dim': 12,
        'season_embedding_dim': 8,
        'num_epochs': 300,
        'batch_size': 64,
        'learning_rate': 0.0001,
        'dropout': 0.5,
        'patience': 30,
        'normalize_per_station': True,
        'use_seasons': True,
        'use_old': False,
    },
    
    # NO DROPOUT EXPERIMENTS
    {
        'name': 'no_dropout_small',
        'features': feature_sets['basic'],
        'sequence_length': 30,
        'prediction_days': 7,
        'hidden_size': 128,
        'num_layers': 2,
        'station_embedding_dim': 8,
        'season_embedding_dim': 4,
        'num_epochs': 100,
        'batch_size': 256,
        'learning_rate': 0.0001,
        'dropout': 0,
        'patience': 10,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    {
        'name': 'no_dropout_large',
        'features': feature_sets['weather_full'],
        'sequence_length': 60,
        'prediction_days': 7,
        'hidden_size': 512,
        'num_layers': 4,
        'station_embedding_dim': 12,
        'season_embedding_dim': 4,
        'num_epochs': 150,
        'batch_size': 64,
        'learning_rate': 0.00001,
        'dropout': 0,
        'patience': 10,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    
    # HEAVY DROPOUT EXPERIMENTS
    {
        'name': 'heavy_dropout',
        'features': feature_sets['comprehensive'],
        'sequence_length': 90,
        'prediction_days': 7,
        'hidden_size': 512,
        'num_layers': 5,
        'station_embedding_dim': 12,
        'season_embedding_dim': 8,
        'num_epochs': 400,
        'batch_size': 64,
        'learning_rate': 0.001,
        'dropout': 0.7,
        'patience': 30,
        'normalize_per_station': True,
        'use_seasons': True,
        'use_old': False,
    },
    
    # SINGLE LAYER EXPERIMENTS
    {
        'name': 'single_layer_wide',
        'features': feature_sets['standard'],
        'sequence_length': 40,
        'prediction_days': 7,
        'hidden_size': 1024,
        'num_layers': 1,
        'station_embedding_dim': 12,
        'season_embedding_dim': 4,
        'num_epochs': 250,
        'batch_size': 128,
        'learning_rate': 0.001,
        'dropout': 0.2,
        'patience': 20,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    
    # DEEP NARROW EXPERIMENTS
    {
        'name': 'deep_narrow',
        'features': feature_sets['standard'],
        'sequence_length': 60,
        'prediction_days': 7,
        'hidden_size': 128,
        'num_layers': 7,
        'station_embedding_dim': 8,
        'season_embedding_dim': 4,
        'num_epochs': 400,
        'batch_size': 64,
        'learning_rate': 0.00001,
        'dropout': 0.5,
        'patience': 30,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    
    # EXTREME SEQUENCE LENGTH
    {
        'name': 'extreme_sequence',
        'features': feature_sets['extended_water'],
        'sequence_length': 120,
        'prediction_days': 7,
        'hidden_size': 512,
        'num_layers': 4,
        'station_embedding_dim': 12,
        'season_embedding_dim': 4,
        'num_epochs': 300,
        'batch_size': 32,
        'learning_rate': 0.0001,
        'dropout': 0.4,
        'patience': 30,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    
    # FEATURE-SPECIFIC OPTIMIZED
    {
        'name': 'water_optimized',
        'features': feature_sets['extended_water'],
        'sequence_length': 40,
        'prediction_days': 7,
        'hidden_size': 256,
        'num_layers': 3,
        'station_embedding_dim': 8,
        'season_embedding_dim': 4,
        'num_epochs': 300,
        'batch_size': 128,
        'learning_rate': 0.0001,
        'dropout': 0.2,
        'patience': 30,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    {
        'name': 'temperature_optimized',
        'features': feature_sets['temperature_focus'],
        'sequence_length': 40,
        'prediction_days': 7,
        'hidden_size': 256,
        'num_layers': 3,
        'station_embedding_dim': 12,
        'season_embedding_dim': 4,
        'num_epochs': 300,
        'batch_size': 64,
        'learning_rate': 0.0001,
        'dropout': 0.4,
        'patience': 30,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
    {
        'name': 'pressure_optimized',
        'features': feature_sets['pressure_focus'],
        'sequence_length': 90,
        'prediction_days': 7,
        'hidden_size': 256,
        'num_layers': 3,
        'station_embedding_dim': 12,
        'season_embedding_dim': 4,
        'num_epochs': 400,
        'batch_size': 64,
        'learning_rate': 0.00001,
        'dropout': 0.4,
        'patience': 30,
        'normalize_per_station': True,
        'use_seasons': False,
        'use_old': False,
    },
]

# Function to generate all experiments with date ranges
def generate_all_experiments():
    """
    Generate all experiments by combining base experiments with date ranges
    """
    all_experiments = []
    
    print("================================")
    print("ALL EXPERIMENTS TO BE RAN:")
    exp_model_no = 1
    for base_exp in base_experiments:
        for date_range_train in date_ranges_train:
            # Create a copy of the base experiment
            exp = base_exp.copy()
            
            # Update the date range
            if date_range_train is None:
                exp['date_range_train'] = None
                exp['last_n_days'] = 0  # Use all data
                date_str = 'all_data'
            else:
                exp['date_range_train'] = date_range_train
                exp['last_n_days'] = 0
                date_str = f'last_{abs(date_range_train[0])}_days'
                
            for date_range_test in date_ranges_test:    
                exp_sub = exp.copy()   
                if date_range_test is None:
                    exp_sub['date_range_test'] = None
                    date_str_test = 'all_data'
                else:
                    exp_sub['date_range_test'] = date_range_test
                    date_str_test = f'range_{date_range_test[0]}-{date_range_test[1]}_days'
            
                # Update the experiment name
                exp_sub['name'] = f"{base_exp['name']}_{date_str}_{date_str_test}"
                exp_sub['model_name'] = f"{exp_model_no}_{base_exp['name']}_{date_str}"
                print(f"{exp_sub['name']}\n" )
                all_experiments.append(exp_sub)
            exp_model_no += 1
    
    
    print("================================")
    return all_experiments, base_experiments, date_ranges_train