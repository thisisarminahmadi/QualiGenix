"""
QualiGenix: Data Integration and EDA
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class PharmDataIntegrator:
    """
    Data integration and EDA for pharmaceutical manufacturing data
    """
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.master_df = None
        self.normalization_df = None
        self.laboratory_df = None
        self.process_df = None
        self.time_series_files = []
        
    def load_base_datasets(self):
        """Load the main CSV files"""
        logger.info("Loading base datasets...")
        
        # Load normalization data
        norm_file = self.data_path / "Normalization.csv"
        if norm_file.exists():
            self.normalization_df = pd.read_csv(norm_file, delimiter=';')
            logger.info(f"Loaded normalization data: {self.normalization_df.shape}")
        
        # Load laboratory data
        lab_file = self.data_path / "Laboratory.csv"
        if lab_file.exists():
            self.laboratory_df = pd.read_csv(lab_file, delimiter=';')
            logger.info(f"Loaded laboratory data: {self.laboratory_df.shape}")
        
        # Load process data
        proc_file = self.data_path / "Process.csv"
        if proc_file.exists():
            self.process_df = pd.read_csv(proc_file, delimiter=';')
            logger.info(f"Loaded process data: {self.process_df.shape}")
        
        return self
    
    def load_time_series_metadata(self):
        """Load metadata from time series files without loading full data"""
        logger.info("Scanning time series files...")
        
        process_dir = self.data_path / "Process"
        if process_dir.exists():
            self.time_series_files = list(process_dir.glob("*.csv"))
            logger.info(f"Found {len(self.time_series_files)} time series files")
            
            # Sample a few files to understand structure
            time_series_metadata = []
            for file in self.time_series_files[:3]:  # Sample first 3 files
                try:
                    sample_df = pd.read_csv(file, delimiter=';', nrows=10)
                    batch_num = int(file.stem)
                    time_series_metadata.append({
                        'batch': batch_num,
                        'file': file.name,
                        'columns': list(sample_df.columns),
                        'sample_shape': sample_df.shape
                    })
                except Exception as e:
                    logger.warning(f"Could not read {file}: {e}")
            
            self.time_series_metadata = time_series_metadata
            logger.info(f"Time series columns: {time_series_metadata[0]['columns'] if time_series_metadata else 'None'}")
        
        return self
    
    def create_master_dataset(self):
        """Merge all base datasets into a master dataset"""
        logger.info("Creating master dataset...")
        
        # Start with laboratory data as base
        if self.laboratory_df is not None:
            master = self.laboratory_df.copy()
            logger.info(f"Base dataset (Laboratory): {master.shape}")
        else:
            raise ValueError("Laboratory data is required as base dataset")
        
        # Merge with process data
        if self.process_df is not None:
            # Merge on batch column
            master = master.merge(
                self.process_df, 
                on=['batch', 'code'], 
                how='left',
                suffixes=('', '_process')
            )
            logger.info(f"After merging process data: {master.shape}")
        
        # Merge with normalization data
        if self.normalization_df is not None:
            # Rename columns to match
            norm_df = self.normalization_df.copy()
            norm_df.rename(columns={'Product code': 'code'}, inplace=True)
            master = master.merge(
                norm_df,
                on='code',
                how='left',
                suffixes=('', '_norm')
            )
            logger.info(f"After merging normalization data: {master.shape}")
        
        self.master_df = master
        logger.info(f"Master dataset created with shape: {master.shape}")
        
        return self
    
    def clean_data(self):
        """Clean and preprocess the master dataset"""
        logger.info("Cleaning data...")
        
        if self.master_df is None:
            raise ValueError("Master dataset not created yet")
        
        # Handle missing values
        missing_summary = self.master_df.isnull().sum()
        logger.info(f"Columns with missing values: {missing_summary[missing_summary > 0].shape[0]}")
        
        # Convert data types
        numeric_columns = self.master_df.select_dtypes(include=[np.number]).columns
        logger.info(f"Numeric columns: {len(numeric_columns)}")
        
        # Remove duplicates
        initial_shape = self.master_df.shape
        self.master_df = self.master_df.drop_duplicates()
        logger.info(f"Removed {initial_shape[0] - self.master_df.shape[0]} duplicate rows")
        
        return self
    
    def generate_eda_summary(self):
        """Generate comprehensive EDA summary"""
        logger.info("Generating EDA summary...")
        
        if self.master_df is None:
            raise ValueError("Master dataset not created yet")
        
        # Basic statistics
        eda_summary = {
            'dataset_info': {
                'shape': self.master_df.shape,
                'columns': list(self.master_df.columns),
                'dtypes': self.master_df.dtypes.value_counts().to_dict(),
                'memory_usage': f"{self.master_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            },
            'missing_values': self.master_df.isnull().sum().to_dict(),
            'numeric_summary': self.master_df.describe().to_dict(),
            'target_variables': {
                'dissolution_av': self.master_df['dissolution_av'].describe().to_dict() if 'dissolution_av' in self.master_df.columns else None,
                'dissolution_min': self.master_df['dissolution_min'].describe().to_dict() if 'dissolution_min' in self.master_df.columns else None,
                'batch_yield': self.master_df['batch_yield'].describe().to_dict() if 'batch_yield' in self.master_df.columns else None,
                'impurities_total': self.master_df['impurities_total'].describe().to_dict() if 'impurities_total' in self.master_df.columns else None,
            }
        }
        
        self.eda_summary = eda_summary
        logger.info("EDA summary generated")
        
        return eda_summary
    
    def create_correlation_analysis(self):
        """Create correlation analysis for key variables"""
        logger.info("Creating correlation analysis...")
        
        # Select numeric columns for correlation
        numeric_df = self.master_df.select_dtypes(include=[np.number])
        
        # Target variables
        target_cols = ['dissolution_av', 'dissolution_min', 'batch_yield', 'impurities_total']
        available_targets = [col for col in target_cols if col in numeric_df.columns]
        
        if available_targets:
            # Calculate correlations with target variables
            correlations = {}
            for target in available_targets:
                corr_with_target = numeric_df.corr()[target].abs().sort_values(ascending=False)
                correlations[target] = corr_with_target.head(10).to_dict()
            
            self.correlations = correlations
            logger.info(f"Correlation analysis completed for targets: {available_targets}")
        
        return self
    
    def save_master_dataset(self, output_path: str = "data/processed/Master.csv"):
        """Save the master dataset"""
        logger.info(f"Saving master dataset to {output_path}...")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.master_df.to_csv(output_file, index=False)
        logger.info(f"Master dataset saved: {self.master_df.shape}")
        
        return output_file
    
    def generate_visualizations(self, output_dir: str = "data/processed/plots"):
        """Generate key visualizations"""
        logger.info("Generating visualizations...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Target variable distributions
        target_cols = ['dissolution_av', 'dissolution_min', 'batch_yield', 'impurities_total']
        available_targets = [col for col in target_cols if col in self.master_df.columns]
        
        if available_targets:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(available_targets[:4]):
                if i < len(axes):
                    self.master_df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(output_path / 'target_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Correlation heatmap
        numeric_df = self.master_df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            # Select subset of columns for readability
            key_columns = available_targets + ['api_water', 'tbl_speed_mean', 'main_CompForce mean', 'batch_yield']
            key_columns = [col for col in key_columns if col in numeric_df.columns][:15]
            
            if len(key_columns) > 1:
                plt.figure(figsize=(12, 10))
                correlation_matrix = numeric_df[key_columns].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
                plt.title('Correlation Matrix - Key Variables')
                plt.tight_layout()
                plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Visualizations saved to {output_path}")
        
        return output_path


def main():
    """Main execution function"""
    logger.info("Starting QualiGenix Phase 1 - Data Integration & EDA")
    
    # Initialize integrator
    integrator = PharmDataIntegrator(data_path="/Users/armin/Desktop/BMS_Demo_Toy/data")
    
    # Execute Phase 1 pipeline
    try:
        # Load all datasets
        integrator.load_base_datasets()
        integrator.load_time_series_metadata()
        
        # Create master dataset
        integrator.create_master_dataset()
        integrator.clean_data()
        
        # Generate EDA
        eda_summary = integrator.generate_eda_summary()
        integrator.create_correlation_analysis()
        
        # Save results
        master_file = integrator.save_master_dataset()
        plots_dir = integrator.generate_visualizations()
        
        # Print summary
        print("\n" + "="*60)
        print("QualiGenix Phase 1 - COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Master Dataset: {integrator.master_df.shape}")
        print(f"Saved to: {master_file}")
        print(f"Plots saved to: {plots_dir}")
        print("\nTarget Variables Available:")
        for target, stats in eda_summary['target_variables'].items():
            if stats:
                print(f"  - {target}: μ={stats['mean']:.2f}, σ={stats['std']:.2f}")
        
        print(f"\nTime Series Files: {len(integrator.time_series_files)} batches")
        print("Phase 1 Complete! Ready for Phase 2 (ML Modeling)")
        
        return integrator
        
    except Exception as e:
        logger.error(f"Phase 1 failed: {e}")
        raise


if __name__ == "__main__":
    integrator = main() 