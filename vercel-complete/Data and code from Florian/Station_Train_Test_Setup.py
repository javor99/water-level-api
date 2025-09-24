from config_loader import load_config, load_mappings


def get_train_test_stations():
    STATION_MAPPINGS = load_mappings()
    print(STATION_MAPPINGS)

    CONFIG = load_config()
    print(CONFIG['base_path_data'])
    base_path_data = CONFIG['base_path_data']

    water_stations = []
    station_files = []

    base_path = f"{base_path_data}\\1_combined_weather_water_level\\"

    for weather_station, water_stations_names in STATION_MAPPINGS.items():
        for water_station_name in water_stations_names:
            water_stations.append(water_station_name)
            station_files.append(f"{base_path}preprocessed_{weather_station}_{water_station_name}.csv")
            
    test_stations = ['Sengelose_test', 'Hove', 'Vaerebro', 'Gudena', 'Vorgod']
    train_stations = []
    train_island = ["Sengelose_train", "Borup", "Himmelev", "Ledreborg"]
    train_jylland = [
        "Mollebaek",
        "Bygholm",
        "Gesager",
        "Skjern",
        "Kirkea",
        "Hoven",
        "Karstoft"
    ]    
    
    # for water_station in water_stations:
    #     if water_station not in test_stations:
    #         train_stations.append(water_station)

    # for water_station in water_stations:
    #     if water_station not in test_stations and water_station in train_island:
    #         train_stations.append(water_station)
            
    for water_station in water_stations:
        if water_station not in test_stations and water_station in train_jylland:
            train_stations.append(water_station)

    return station_files, train_stations, test_stations