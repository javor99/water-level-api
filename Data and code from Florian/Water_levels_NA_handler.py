import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class WaterLevelPreprocessor:
    """
    Preprocess water level data by finding and keeping the longest continuous period
    """
    
    def __init__(self, df, station_name, date_col='observed', water_level_col='water_level'):
        """
        Parameters:
        -----------
        df : pandas DataFrame
            Raw water level data
        date_col : str
            Name of the date column
        water_level_col : str
            Name of the water level column
        """
        self.df = df.copy()
        self.date_col = date_col
        self.water_level_col = water_level_col
        self.station_name = station_name
        
        # Ensure date column is datetime and set as index
        if self.date_col in self.df.columns:
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
            self.df = self.df.set_index(self.date_col)
        
        # Sort by date
        self.df = self.df.sort_index()
        
    def analyze_continuity(self, min_gap_days=7, plot=True):
        """
        Analyze continuous periods in the data
        
        Parameters:
        -----------
        min_gap_days : int
            Minimum gap size (in days) to consider as a break in continuity
        plot : bool
            Whether to plot the analysis
            
        Returns:
        --------
        continuous_periods : list of dict
            List of continuous periods with their statistics
        """
        # Find all non-NA indices
        valid_mask_water_levels = ~self.df[self.water_level_col].isna()
        valid_dates_water_levels = self.df[valid_mask_water_levels].index
        
        if len(valid_dates_water_levels) == 0:
            print("No valid water level data found!")
            return []
        
        # Find continuous periods
        continuous_periods = []
        current_period_start = valid_dates_water_levels[0]
        last_date = valid_dates_water_levels[0]
        
        for date in valid_dates_water_levels[1:]:
            # Check if there's a gap
            gap_days = (date - last_date).days
            
            if gap_days > min_gap_days:
                # End current period
                period_data = self.df.loc[current_period_start:last_date, self.water_level_col]
                valid_data = period_data.dropna()
                
                continuous_periods.append({
                    'start': current_period_start,
                    'end': last_date,
                    'duration_days': (last_date - current_period_start).days + 1,
                    'valid_points': len(valid_data),
                    'total_points': len(period_data),
                    'completeness': len(valid_data) / len(period_data) * 100 if len(period_data) > 0 else 0
                })
                
                # Start new period
                current_period_start = date
            
            last_date = date
        
        # Don't forget the last period
        period_data = self.df.loc[current_period_start:last_date, self.water_level_col]
        valid_data = period_data.dropna()
        
        continuous_periods.append({
            'start': current_period_start,
            'end': last_date,
            'duration_days': (last_date - current_period_start).days + 1,
            'valid_points': len(valid_data),
            'total_points': len(period_data),
            'completeness': len(valid_data) / len(period_data) * 100 if len(period_data) > 0 else 0
        })
        
        # Sort by number of valid points (descending)
        continuous_periods.sort(key=lambda x: x['valid_points'], reverse=True)
        
        # Print analysis
        print(f"Continuity Analysis (gap threshold: {min_gap_days} days)")
        print(f"Total periods found: {len(continuous_periods)}")
        print(f"\nTop 5 continuous periods by data points:")
        print("-" * 80)
        print(f"{'Rank':<5} {'Start':<12} {'End':<12} {'Days':<8} {'Points':<8} {'Complete':<10}")
        print("-" * 80)
        
        for i, period in enumerate(continuous_periods[:5]):
            print(f"{i+1:<5} {period['start'].strftime('%Y-%m-%d'):<12} "
                  f"{period['end'].strftime('%Y-%m-%d'):<12} "
                  f"{period['duration_days']:<8} {period['valid_points']:<8} "
                  f"{period['completeness']:<10.1f}%")
        
        if plot:
            self._plot_continuity_analysis(continuous_periods)
            
        return continuous_periods
    
    def find_best_continuous_period(self, min_days=365, min_completeness=90, 
                                   interpolate_small_gaps=True, max_interpolate_days=3):
        """
        Find the best continuous period of water level data
        
        Parameters:
        -----------
        min_days : int
            Minimum period length in days to consider
        min_completeness : float
            Minimum data completeness percentage required
        interpolate_small_gaps : bool
            Whether to interpolate small gaps within selected period
        max_interpolate_days : int
            Maximum gap size to interpolate
            
        Returns:
        --------
        best_data : DataFrame
            The best continuous period of data
        period_info : dict
            Information about the selected period
        """
        # Get continuous periods
        periods = self.analyze_continuity(min_gap_days=max_interpolate_days + 1)
        
        # Filter by criteria
        suitable_periods = [
            p for p in periods 
            if p['duration_days'] >= min_days and p['completeness'] >= min_completeness
        ]
        
        if not suitable_periods:
            print(f"\nNo periods found meeting criteria (min {min_days} days, "
                  f"{min_completeness}% complete)")
            print("Relaxing completeness requirement...")
            
            # Try with relaxed completeness
            suitable_periods = [
                p for p in periods 
                if p['duration_days'] >= min_days and p['completeness'] >= 70
            ]
        
        if not suitable_periods:
            print("Still no suitable periods. Returning longest available period.")
            best_period = periods[0] if periods else None
        else:
            # Select the period with most valid data points
            best_period = suitable_periods[0]
        
        if best_period is None:
            return pd.DataFrame(), {}
        
        # Extract the best period
        best_data = self.df.loc[best_period['start']:best_period['end']].copy()
        
        # Interpolate small gaps if requested
        if interpolate_small_gaps:
            before_interp = best_data[self.water_level_col].isna().sum()
            best_data[self.water_level_col] = best_data[self.water_level_col].interpolate(
                method='linear',
                limit=max_interpolate_days
            )
            after_interp = best_data[self.water_level_col].isna().sum()
            
            print(f"\nInterpolation: Filled {before_interp - after_interp} small gaps "
                  f"(<= {max_interpolate_days} days)")
        
        # Final cleanup - remove any remaining NAs at the edges
        first_valid = best_data[self.water_level_col].first_valid_index()
        last_valid = best_data[self.water_level_col].last_valid_index()
        
        if first_valid and last_valid:
            best_data = best_data.loc[first_valid:last_valid]
        
        # Calculate final statistics
        period_info = {
            'start': best_data.index[0],
            'end': best_data.index[-1],
            'duration_days': (best_data.index[-1] - best_data.index[0]).days + 1,
            'total_points': len(best_data),
            'valid_points': best_data[self.water_level_col].notna().sum(),
            'completeness': best_data[self.water_level_col].notna().sum() / len(best_data) * 100,
            'mean_level': best_data[self.water_level_col].mean(),
            'std_level': best_data[self.water_level_col].std(),
            'min_level': best_data[self.water_level_col].min(),
            'max_level': best_data[self.water_level_col].max()
        }
        
        print(f"\nSelected Period Summary:")
        print(f"Period: {period_info['start'].strftime('%Y-%m-%d')} to "
              f"{period_info['end'].strftime('%Y-%m-%d')}")
        print(f"Duration: {period_info['duration_days']} days "
              f"({period_info['duration_days']/365:.1f} years)")
        print(f"Data points: {period_info['valid_points']} / {period_info['total_points']} "
              f"({period_info['completeness']:.1f}% complete)")
        print(f"Water level range: {period_info['min_level']:.2f} to "
              f"{period_info['max_level']:.2f} (mean: {period_info['mean_level']:.2f})")
        
        return best_data, period_info
    
    def _plot_continuity_analysis(self, continuous_periods):
        """
        Visualize the continuous periods
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot 1: Full data with continuous periods highlighted
        ax1 = axes[0]
        
        # Plot all data in grey
        ax1.scatter(self.df.index, self.df[self.water_level_col], 
                   c='lightgray', s=1, alpha=0.5, label='All data')
        
        # Highlight top 5 continuous periods
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, period in enumerate(continuous_periods[:5]):
            period_data = self.df.loc[period['start']:period['end'], self.water_level_col]
            ax1.plot(period_data.index, period_data, 
                    color=colors[i], linewidth=2, alpha=0.8,
                    label=f"Period {i+1}: {period['valid_points']} points")
        
        ax1.set_title('Water Level Data - Top 5 Continuous Periods')
        ax1.set_ylabel('Water Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Period lengths
        ax2 = axes[1]
        period_lengths = [p['valid_points'] for p in continuous_periods]
        period_numbers = range(1, len(continuous_periods) + 1)
        
        bars = ax2.bar(period_numbers[:20], period_lengths[:20])  # Show top 20
        bars[0].set_color('green')  # Highlight the best
        
        ax2.set_title('Valid Data Points per Continuous Period (Top 20)')
        ax2.set_xlabel('Period Rank')
        ax2.set_ylabel('Number of Valid Points')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Timeline of periods
        ax3 = axes[2]
        
        for i, period in enumerate(continuous_periods[:10]):  # Top 10
            y_pos = i
            start_date = period['start']
            duration = period['duration_days']
            
            # Draw horizontal bar
            ax3.barh(y_pos, duration, left=start_date, height=0.8,
                    color=colors[i % len(colors)], alpha=0.7)
            
            # Add text with number of points
            ax3.text(start_date + timedelta(days=duration/2), y_pos,
                    f"{period['valid_points']} pts",
                    ha='center', va='center', fontsize=8)
        
        ax3.set_ylim(-0.5, 9.5)
        ax3.set_yticks(range(10))
        ax3.set_yticklabels([f"Period {i+1}" for i in range(10)])
        ax3.set_xlabel('Date')
        ax3.set_title('Timeline of Top 10 Continuous Periods')
        ax3.grid(True, alpha=0.3, axis='x')
        
        plt.title(self.station_name)
        plt.tight_layout()
        plt.show()
    
    def plot_before_after(self, processed_data):
        """
        Plot comparison of original vs processed data
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Original data
        ax1 = axes[0]
        ax1.scatter(self.df.index, self.df[self.water_level_col], 
                   c='blue', s=1, alpha=0.6)
        ax1.set_title(f'Original Data ({len(self.df)} points, '
                     f'{self.df[self.water_level_col].notna().sum()} valid)')
        ax1.set_ylabel('Water Level')
        ax1.grid(True, alpha=0.3)
        
        # Processed data
        ax2 = axes[1]
        ax2.plot(processed_data.index, processed_data[self.water_level_col], 
                'green', linewidth=1, alpha=0.8)
        ax2.set_title(f'Selected Continuous Period ({len(processed_data)} points, '
                     f'{processed_data[self.water_level_col].notna().sum()} valid)')
        ax2.set_ylabel('Water Level')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # Highlight the selected period in the original plot
        if len(processed_data) > 0:
            ax1.axvspan(processed_data.index[0], processed_data.index[-1], 
                       alpha=0.2, color='green', label='Selected period')
            ax1.legend()
        
        plt.title(self.station_name)
        plt.tight_layout()
        plt.show()


def preprocess_water_station(filepath, station_name, output_file_path, min_years=2, 
                           save_processed=True, max_interpolate_days = 5):
    """
    Complete preprocessing pipeline for a water station
    
    Parameters:
    -----------
    filepath : str
        Path to the raw water level CSV
    station_name : str
        Name of the station
    min_years : float
        Minimum years of continuous data required
    save_processed : bool
        Whether to save the processed data
    output_dir : str
        Directory to save processed files
    """
    import os
    
    print(f"\nProcessing station: {station_name}")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Initialize preprocessor
    preprocessor = WaterLevelPreprocessor(df, station_name)
    
    # Find best continuous period
    best_data, period_info = preprocessor.find_best_continuous_period(
        min_days=int(min_years * 365),
        min_completeness=90,
        interpolate_small_gaps=True,
        max_interpolate_days=max_interpolate_days
    )
    
    # Plot comparison
    preprocessor.plot_before_after(best_data)
    
    # Save if requested
    if save_processed and len(best_data) > 0:
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file_path)
        if output_dir:  # Only create if there's a directory path
            os.makedirs(output_dir, exist_ok=True)
        
        best_data.to_csv(output_file_path)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, f"{station_name}_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Station: {station_name}\n")
            f.write(f"Original file: {filepath}\n")
            f.write(f"Processing date: {datetime.now()}\n")
            f.write(f"\nPeriod selected:\n")
            for key, value in period_info.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"\nSaved processed data to: {output_file_path}")
        print(f"Saved metadata to: {metadata_path}")
    
    return best_data, period_info