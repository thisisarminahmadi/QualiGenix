"""
QualiGenix ML Modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

class PharmMLModeler:
    """
    Focused ML modeling for pharmaceutical CQA prediction
    """
    
    def __init__(self, data_path: str = "data/processed/Master.csv"):
        self.data_path = Path(data_path)
        self.master_df = None
        self.models = {}
        self.feature_importance = {}
        self.results = {}
        self.optimized_models = {}  # Initialize here to fix the error
        
        # Target variables (CQAs) - Final product quality attributes
        self.targets = {
            'dissolution_av': 'Average Drug Release (%)',
            'dissolution_min': 'Minimum Drug Release (%)', 
            'batch_yield': 'Batch Yield (%)',
            'impurities_total': 'Total Impurities (%)',
            'impurity_o': 'Impurity O (%)',
            'impurity_l': 'Impurity L (%)',
            'resodual_solvent': 'Residual Solvent (%)'
        }
        
        # Feature categories for pharmaceutical manufacturing
        self.feature_categories = {
            'raw_materials': [
                'api_water', 'api_total_impurities', 'api_content', 'api_ps01', 'api_ps05', 'api_ps09',
                'lactose_water', 'lactose_sieve0045', 'lactose_sieve015', 'lactose_sieve025',
                'smcc_water', 'smcc_td', 'smcc_bd', 'smcc_ps01', 'smcc_ps05', 'smcc_ps09',
                'starch_ph', 'starch_water'
            ],
            'process_parameters': [
                'tbl_speed_mean', 'tbl_speed_change', 'fom_mean', 'fom_change',
                'SREL_startup_mean', 'SREL_production_mean', 'SREL_production_max',
                'main_CompForce mean', 'main_CompForce_sd', 'pre_CompForce_mean',
                'tbl_fill_mean', 'tbl_fill_sd', 'stiffness_mean', 'stiffness_max', 'stiffness_min',
                'ejection_mean', 'ejection_max', 'ejection_min'
            ],
            'intermediate_quality': [
                'tbl_rsd_weight', 'fct_rsd_weight', 'tbl_av_hardness', 'fct_av_hardness',
                'tbl_tensile', 'fct_tensile', 'tbl_yield'
            ],
            'genealogy': [
                'api_batch', 'smcc_batch', 'lactose_batch', 'starch_batch', 
                'code', 'strength', 'size'
            ]
        }
        
    def load_master_data(self):
        """Load the master dataset"""
        logging.info(f"Loading master dataset from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Master dataset not found at {self.data_path}")
        
        self.master_df = pd.read_csv(self.data_path)
        logging.info(f"Loaded dataset with shape: {self.master_df.shape}")
        
        return self
    
    def prepare_features(self):
        """Prepare features for ML modeling with proper missing value handling"""
        logging.info("Preparing features for ML modeling...")
        
        if self.master_df is None:
            raise ValueError("Master dataset not loaded")
        
        # Create a copy for feature engineering
        df = self.master_df.copy()
        
        # Handle categorical variables (genealogy)
        categorical_cols = ['api_batch', 'smcc_batch', 'lactose_batch', 'starch_batch', 'code', 'strength', 'size']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        # Feature engineering - create pharmaceutical-specific features
        df['compression_ratio'] = df['main_CompForce mean'] / (df['tbl_fill_mean'] + 1e-6)
        df['speed_efficiency'] = df['tbl_speed_mean'] / (df['total_waste'] + 1)
        df['hardness_consistency'] = df['tbl_max_hardness'] - df['tbl_min_hardness']
        df['weight_consistency'] = df['tbl_max_weight'] - df['tbl_min_weight']
        df['thickness_consistency'] = df['tbl_max_thickness'] - df['tbl_min_thickness']
        
        # Process stability metrics
        df['srel_stability'] = df['SREL_production_max'] - df['SREL_production_mean']
        df['force_stability'] = df['main_CompForce_sd'] / (df['main_CompForce mean'] + 1e-6)
        
        # Normalize batch size impact
        if 'Batch Size (tablets)' in df.columns:
            df['normalized_yield'] = df['batch_yield'] / df['Batch Size (tablets)'] * 1000
        
        # Handle missing values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        self.processed_df = df
        logging.info(f"Feature preparation completed. Shape: {df.shape}")
        logging.info(f"Missing values after processing: {df.isnull().sum().sum()}")
        
        return self
    
    def select_features(self, target_col, method='mutual_info', k=30):
        """Select best features for modeling"""
        logging.info(f"Selecting features for target: {target_col}")
        
        if self.processed_df is None:
            raise ValueError("Processed dataset not available")
        
        # Prepare data
        X = self.processed_df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
        y = self.processed_df[target_col].dropna()
        
        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Remove columns with too many missing values
        X = X.dropna(axis=1, thresh=len(X) * 0.7)
        
        # Final NaN check and imputation
        if X.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='median')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        
        # Feature selection
        selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.selected_features = selected_features
        logging.info(f"Selected {len(selected_features)} features for {target_col}")
        
        return selected_features
    
    def create_proper_splits(self, X, y, test_size=0.2, val_size=0.2):
        """Create proper 60/20/20 train/validation/test splits"""
        # First split: 80% train+val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        
        # Second split: 60% train, 20% validation from the 80%
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=None
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_single_model(self, target_col, model_name, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train a single model for a target"""
        logging.info(f"Training {model_name} for {target_col}")
        
        # Initialize model
        if model_name == 'LightGBM':
            model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                num_leaves=50,
                random_state=42,
                verbose=-1
            )
            # Train model with proper LightGBM syntax
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )
        elif model_name == 'CatBoost':
            model = cb.CatBoostRegressor(
                iterations=200,
                depth=8,
                learning_rate=0.1,
                random_seed=42,
                verbose=False
            )
            # Train model with proper CatBoost syntax
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train': {
                'r2': r2_score(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'mae': mean_absolute_error(y_train, y_pred_train)
            },
            'val': {
                'r2': r2_score(y_val, y_pred_val),
                'rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'mae': mean_absolute_error(y_val, y_pred_val)
            },
            'test': {
                'r2': r2_score(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test)
            }
        }
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(self.selected_features, model.feature_importances_))
        elif hasattr(model, 'get_feature_importance'):
            feature_importance = dict(zip(self.selected_features, model.get_feature_importance()))
        else:
            feature_importance = {}
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'predictions': {
                'train': y_pred_train,
                'val': y_pred_val,
                'test': y_pred_test
            },
            'actual': {
                'train': y_train,
                'val': y_val,
                'test': y_test
            }
        }
    
    def train_models_parallel(self, target_col):
        """Train models in parallel for efficiency"""
        logging.info(f"Training models for target: {target_col}")
        
        if self.processed_df is None:
            raise ValueError("Processed dataset not available")
        
        # Prepare data
        X = self.processed_df[self.selected_features]
        y = self.processed_df[target_col].dropna()
        
        # Align data
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Final NaN check
        if X.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='median')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        
        # Create proper splits (60/20/20)
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_proper_splits(X, y)
        
        logging.info(f"Data splits - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        # Train models in parallel
        models_to_train = ['LightGBM', 'CatBoost']
        results = {}
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit all training tasks
            future_to_model = {
                executor.submit(
                    self.train_single_model, target_col, model_name, 
                    X_train, X_val, X_test, y_train, y_val, y_test
                ): model_name for model_name in models_to_train
            }
            
            # Collect results
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    results[model_name] = result
                    logging.info(f"Completed training {model_name} for {target_col}")
                except Exception as e:
                    logging.error(f"Error training {model_name} for {target_col}: {e}")
        
        self.results[target_col] = results
        logging.info(f"Model training completed for {target_col}")
        
        return results
    
    def save_models(self, output_dir: str = "data/processed/models"):
        """Save trained models for deployment"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for target_col, models in self.results.items():
            for model_name, result in models.items():
                model_file = output_path / f"{target_col}_{model_name}.joblib"
                joblib.dump(result['model'], model_file)
                logging.info(f"Saved model: {model_file}")
        
        # Save feature lists
        for target_col in self.results.keys():
            features_file = output_path / f"{target_col}_features.txt"
            with open(features_file, 'w') as f:
                f.write('\n'.join(self.selected_features))
        
        return output_path
    
    def generate_model_report(self, output_dir: str = "data/processed/ml_results"):
        """Generate comprehensive model report"""
        logging.info("Generating model report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create model comparison plots
        for target_col in self.targets.keys():
            if target_col in self.results:
                self._plot_model_comparison(target_col, output_path)
                self._plot_feature_importance(target_col, output_path)
                self._plot_predictions_vs_actual(target_col, output_path)
        
        # Save results summary
        self._save_results_summary(output_path)
        
        logging.info(f"Model report saved to {output_path}")
        return output_path
    
    def _plot_model_comparison(self, target_col, output_path):
        """Plot model comparison"""
        results = self.results[target_col]
        
        models = list(results.keys())
        test_r2_scores = [results[model]['metrics']['test']['r2'] for model in models]
        test_rmse_scores = [results[model]['metrics']['test']['rmse'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² scores
        ax1.bar(models, test_r2_scores, color='skyblue', alpha=0.7)
        ax1.set_title(f'Test R² Scores - {target_col}')
        ax1.set_ylabel('R² Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # RMSE scores
        ax2.bar(models, test_rmse_scores, color='lightcoral', alpha=0.7)
        ax2.set_title(f'Test RMSE Scores - {target_col}')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / f'model_comparison_{target_col}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, target_col, output_path):
        """Plot feature importance"""
        if target_col not in self.results:
            return
        
        results = self.results[target_col]
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        for i, (model_name, result) in enumerate(results.items()):
            if i < 2:
                importance = result['feature_importance']
                # Get top 15 features
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
                features, scores = zip(*top_features)
                
                axes[i].barh(features, scores, color='lightgreen', alpha=0.7)
                axes[i].set_title(f'Feature Importance - {model_name}')
                axes[i].set_xlabel('Importance Score')
        
        plt.tight_layout()
        plt.savefig(output_path / f'feature_importance_{target_col}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_predictions_vs_actual(self, target_col, output_path):
        """Plot predictions vs actual values"""
        results = self.results[target_col]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for i, (model_name, result) in enumerate(results.items()):
            if i < 2:
                actual = result['actual']['test']
                predictions = result['predictions']['test']
                r2 = result['metrics']['test']['r2']
                
                axes[i].scatter(actual, predictions, alpha=0.6)
                axes[i].plot([actual.min(), actual.max()], 
                           [actual.min(), actual.max()], 'r--', lw=2)
                axes[i].set_xlabel('Actual')
                axes[i].set_ylabel('Predicted')
                axes[i].set_title(f'{model_name} - R² = {r2:.3f}')
        
        plt.tight_layout()
        plt.savefig(output_path / f'predictions_vs_actual_{target_col}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results_summary(self, output_path):
        """Save results summary to CSV"""
        summary_data = []
        
        for target_col in self.targets.keys():
            if target_col in self.results:
                for model_name, result in self.results[target_col].items():
                    for split in ['train', 'val', 'test']:
                        summary_data.append({
                            'target': target_col,
                            'model': model_name,
                            'split': split,
                            'r2': result['metrics'][split]['r2'],
                            'rmse': result['metrics'][split]['rmse'],
                            'mae': result['metrics'][split]['mae']
                        })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / 'model_results_summary.csv', index=False)
        
        # Save feature importance
        for target_col in self.targets.keys():
            if target_col in self.results:
                importance_data = []
                for model_name, result in self.results[target_col].items():
                    for feature, score in result['feature_importance'].items():
                        importance_data.append({
                            'target': target_col,
                            'model': model_name,
                            'feature': feature,
                            'importance': score
                        })
                
                importance_df = pd.DataFrame(importance_data)
                importance_df.to_csv(output_path / f'feature_importance_{target_col}.csv', index=False)


def main():
    """Main execution function for ML modeling"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Starting QualiGenix Phase 2 - Focused ML Modeling")
    
    # Initialize modeler
    modeler = PharmMLModeler()
    
    try:
        # Load and prepare data
        modeler.load_master_data()
        modeler.prepare_features()
        
        # Train models for each CQA target
        for target_col in modeler.targets.keys():
            if target_col in modeler.master_df.columns:
                logger.info(f"Training models for {target_col}...")
                
                # Select features
                modeler.select_features(target_col, method='mutual_info', k=30)
                
                # Train models in parallel
                modeler.train_models_parallel(target_col)
        
        # Save models for deployment
        models_dir = modeler.save_models()
        
        # Generate comprehensive report
        results_dir = modeler.generate_model_report()
        
        # Print summary
        print("\n" + "="*60)
        print("QualiGenix Phase 2 - ML MODELING COMPLETED")
        print("="*60)
        print(f"Models saved to: {models_dir}")
        print(f"Results saved to: {results_dir}")
        print("\nModel Performance Summary (Test Set):")
        
        for target_col in modeler.targets.keys():
            if target_col in modeler.results:
                print(f"\n{target_col.upper()}:")
                for model_name, result in modeler.results[target_col].items():
                    test_r2 = result['metrics']['test']['r2']
                    test_rmse = result['metrics']['test']['rmse']
                    print(f"  {model_name}: R² = {test_r2:.3f}, RMSE = {test_rmse:.3f}")
        
        print("\nPhase 2 Complete! Ready for Phase 3 (GenAI Agent)")
        
        return modeler
        
    except Exception as e:
        logger.error(f"Phase 2 failed: {e}")
        raise


if __name__ == "__main__":
    modeler = main() 