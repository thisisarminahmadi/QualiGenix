"""
QualiGenix: GenAI-Powered Decision Intelligence Platform
Streamlit Dashboard for Pharmaceutical Manufacturing
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import joblib
from datetime import datetime
import warnings
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from genai_agent import QualiGenixAgent

# Page configuration
st.set_page_config(
    page_title="QualiGenix: GenAI-Powered Decision Intelligence Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the master dataset"""
    try:
        df = pd.read_csv("data/processed/Master.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_models():
    """Load trained ML models"""
    models = {}
    models_dir = Path("data/processed/models")
    
    if not models_dir.exists():
        return {}
    
    # Load model results summary
    results_file = Path("data/processed/ml_results/model_results_summary.csv")
    if results_file.exists():
        results_df = pd.read_csv(results_file)
        
        # Filter to test set results only
        test_results = results_df[results_df['split'] == 'test']
        
        for _, row in test_results.iterrows():
            model_name = f"{row['target']}_{row['model']}"
            models[model_name] = {
                'target': row['target'],
                'model_type': row['model'],
                'r2_score': row['r2'],
                'rmse': row['rmse'],
                'mae': row['mae']
            }
    
    return models

@st.cache_data
def load_feature_importance():
    """Load feature importance data"""
    importance_data = {}
    results_dir = Path("data/processed/ml_results")
    
    if results_dir.exists():
        for file in results_dir.glob("feature_importance_*.csv"):
            target = file.stem.replace("feature_importance_", "")
            df = pd.read_csv(file)
            importance_data[target] = df
    
    return importance_data

def load_genai_agent():
    """Load the GenAI agent"""
    try:
        return QualiGenixAgent()
    except Exception as e:
        st.error(f"Error loading GenAI agent: {e}")
        return None

class MLPredictor:
    """ML-powered prediction engine using trained models"""
    
    def __init__(self, models_dir="data/processed/models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.features = {}
        self._load_models()
    
    def _load_models(self):
        """Load all trained models and their features"""
        targets = ['dissolution_av', 'dissolution_min', 'batch_yield', 'impurities_total', 
                  'impurity_o', 'impurity_l', 'resodual_solvent']
        
        for target in targets:
            for model_type in ['LightGBM', 'CatBoost']:
                model_file = self.models_dir / f"{target}_{model_type}.joblib"
                if model_file.exists():
                    try:
                        self.models[f"{target}_{model_type}"] = joblib.load(model_file)
                    except Exception as e:
                        st.warning(f"Could not load {model_file}: {e}")
            
            # Load feature list
            features_file = self.models_dir / f"{target}_features.txt"
            if features_file.exists():
                with open(features_file, 'r') as f:
                    self.features[target] = f.read().strip().split('\n')
    
    def predict_single(self, target, parameters):
        """Make prediction for a single target"""
        model_key = f"{target}_CatBoost"  # Use CatBoost by default
        if model_key not in self.models:
            return None
        
        if target not in self.features:
            return None
        
        # Create feature vector in correct order
        feature_vector = []
        for feature in self.features[target]:
            if feature in parameters:
                feature_vector.append(parameters[feature])
            else:
                # Use default values
                if 'water' in feature.lower():
                    feature_vector.append(1.0)
                elif 'force' in feature.lower() or 'compression' in feature.lower():
                    feature_vector.append(4.0)
                elif 'speed' in feature.lower():
                    feature_vector.append(100.0)
                elif 'stiffness' in feature.lower():
                    feature_vector.append(1.5)
                else:
                    feature_vector.append(0.0)
        
        try:
            model = self.models[model_key]
            prediction = model.predict([feature_vector])[0]
            return prediction
        except Exception as e:
            st.error(f"Error predicting {target}: {e}")
            return None
    
    def predict_all(self, parameters):
        """Make predictions for all targets"""
        predictions = {}
        targets = ['dissolution_av', 'dissolution_min', 'batch_yield', 'impurities_total', 
                  'impurity_o', 'impurity_l', 'resodual_solvent']
        
        for target in targets:
            pred = self.predict_single(target, parameters)
            if pred is not None:
                predictions[target] = pred
        
        return predictions

class MLOptimizer:
    """ML-powered optimization engine"""
    
    def __init__(self, predictor):
        self.predictor = predictor
    
    def optimize_for_targets(self, targets, constraints, bounds):
        """Optimize parameters to achieve target values"""
        
        def objective(params):
            # params: [api_water, compression, speed, stiffness, batch_size, strength, size]
            parameters = {
                'api_water': params[0],
                'main_CompForce mean': params[1],
                'tbl_speed_mean': params[2],
                'stiffness_mean': params[3],
                'Batch Size (tablets)': params[4],
                'strength': params[5],
                'size': params[6]
            }
            
            # Get predictions
            predictions = self.predictor.predict_all(parameters)
            
            # Calculate total error from targets
            total_error = 0
            for target, target_value in targets.items():
                if target in predictions:
                    error = abs(predictions[target] - target_value)
                    # Weight different targets
                    if 'dissolution' in target:
                        total_error += error * 2  # Higher weight for dissolution
                    elif 'yield' in target:
                        total_error += error * 1.5
                    else:
                        total_error += error
            
            return total_error
        
        # Use differential evolution for global optimization
        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=15
        )
        
        if result.success:
            optimal_params = {
                'api_water': result.x[0],
                'main_CompForce mean': result.x[1],
                'tbl_speed_mean': result.x[2],
                'stiffness_mean': result.x[3],
                'Batch Size (tablets)': result.x[4],
                'strength': result.x[5],
                'size': result.x[6]
            }
            
            # Get predictions for optimal parameters
            optimal_predictions = self.predictor.predict_all(optimal_params)
            
            return optimal_params, optimal_predictions, result.fun
        else:
            return None, None, None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header"> QualiGenix: GenAI-Powered Decision Intelligence Platform</h1>', unsafe_allow_html=True)
    st.markdown("**Pharmaceutical Manufacturing Quality Prediction & Optimization**")
    
    # Load data and models
    df = load_data()
    models = load_models()
    importance_data = load_feature_importance()
    genai_agent = load_genai_agent()
    
    # Initialize ML components
    ml_predictor = MLPredictor()
    ml_optimizer = MLOptimizer(ml_predictor)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Data Explorer", "AI Predictions", "GenAI Assistant", 
        "Experiment Simulator", "Scenario Generator", "ML Explorer"
    ])
    
    # Tab 1: Data Explorer
    with tab1:
        st.header("Pharmaceutical Manufacturing Data Explorer")
        st.info("📊 **What this tab does:** Explore your historical manufacturing data with interactive visualizations, filters, and statistical analysis to understand quality trends and correlations.")
        
        if df is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Quality metrics overview
                st.subheader("Quality Metrics Overview")
                
                # Key metrics
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    avg_dissolution = df['dissolution_av'].mean()
                    st.metric("Avg Dissolution", f"{avg_dissolution:.1f}%")
                
                with metrics_col2:
                    avg_yield = df['batch_yield'].mean()
                    st.metric("Avg Yield", f"{avg_yield:.1f}%")
                
                with metrics_col3:
                    avg_impurities = df['impurities_total'].mean()
                    st.metric("Avg Impurities", f"{avg_impurities:.3f}%")
                
                with metrics_col4:
                    total_batches = len(df)
                    st.metric("Total Batches", f"{total_batches:,}")
                
                # Interactive filters
                st.subheader("Data Filters")
                col_filter1, col_filter2 = st.columns(2)
                
                with col_filter1:
                    yield_range = st.slider("Batch Yield Range", 
                                          float(df['batch_yield'].min()), 
                                          float(df['batch_yield'].max()), 
                                          (float(df['batch_yield'].min()), 
                                           float(df['batch_yield'].max())))
                
                with col_filter2:
                    dissolution_range = st.slider("Dissolution Range", 
                                                float(df['dissolution_av'].min()), 
                                                float(df['dissolution_av'].max()), 
                                                (float(df['dissolution_av'].min()), 
                                                 float(df['dissolution_av'].max())))
                
                # Filter data
                filtered_df = df[
                    (df['batch_yield'] >= yield_range[0]) & 
                    (df['batch_yield'] <= yield_range[1]) &
                    (df['dissolution_av'] >= dissolution_range[0]) & 
                    (df['dissolution_av'] <= dissolution_range[1])
                ]
                
                st.write(f"Showing {len(filtered_df)} batches (filtered from {len(df)} total)")
                
                # Quality distribution 
                st.subheader("Quality Distribution")
                
                # Create two columns for distribution charts
                dist_col1, dist_col2 = st.columns(2)
                
                with dist_col1:
                    # Dissolution distribution
                    fig_diss = px.histogram(df, x='dissolution_av', nbins=20, 
                                          title="Dissolution Distribution")
                    fig_diss.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_diss, use_container_width=True)
                
                with dist_col2:
                    # Yield distribution
                    fig_yield = px.histogram(df, x='batch_yield', nbins=20, 
                                           title="Yield Distribution")
                    fig_yield.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_yield, use_container_width=True)
            
            with col2:
                # Data quality info
                st.subheader("Dataset Information")
                st.write(f"**Total Batches:** {len(df):,}")
                st.write(f"**Features:** {len(df.columns)}")
                st.write(f"**Date Range:** Historical data")
                
                # Missing values
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                st.write(f"**Data Completeness:** {100-missing_pct:.1f}%")
            
            # Correlation analysis
            st.subheader("Quality Correlations")
            
            # Select numeric columns for correlation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            key_cols = ['dissolution_av', 'dissolution_min', 'batch_yield', 'impurities_total', 
                        'api_water', 'main_CompForce mean', 'tbl_speed_mean', 'stiffness_mean']
            
            # Filter to available columns
            available_cols = [col for col in key_cols if col in numeric_cols]
            
            if len(available_cols) > 1:
                corr_matrix = df[available_cols].corr()
                
                fig_corr = px.imshow(corr_matrix, 
                                   text_auto=True, 
                                   aspect="auto",
                                   title="Quality Metrics Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Process parameter analysis
            st.subheader("Process Parameter Analysis")
            
            param_col1, param_col2 = st.columns(2)
            
            with param_col1:
                # Compression force vs dissolution
                fig_comp = px.scatter(df, x='main_CompForce mean', y='dissolution_av',
                                   title="Compression Force vs Dissolution",
                                   labels={'main_CompForce mean': 'Compression Force (kN)',
                                          'dissolution_av': 'Dissolution (%)'})
                st.plotly_chart(fig_comp, use_container_width=True)
            
            with param_col2:
                # Speed vs yield
                fig_speed = px.scatter(df, x='tbl_speed_mean', y='batch_yield',
                                    title="Tablet Speed vs Yield",
                                    labels={'tbl_speed_mean': 'Speed (rpm)',
                                           'batch_yield': 'Yield (%)'})
                st.plotly_chart(fig_speed, use_container_width=True)
        
        else:
            st.error("Could not load data. Please ensure the Master.csv file exists in data/processed/")
    
    # Tab 2: AI Predictions - FIXED: Remove boxes from predictions
    with tab2:
        st.header("AI-Powered Quality Predictions")
        st.info("🤖 **What this tab does:** Use trained ML models to predict future batch quality (dissolution, yield, impurities) based on your input parameters. Adjust sliders to see how different process conditions affect predicted outcomes.")
        
        if not models:
            st.warning("No trained models found. Please run Phase 2 first.")
        else:
            st.subheader("Input Parameters")
            
            col_input1, col_input2 = st.columns(2)
            
            with col_input1:
                st.markdown("**Process Parameters:**")
                api_water = st.slider("API Water Content", 0.5, 3.0, 1.5, 0.1)
                compression = st.slider("Compression Force", 2.0, 15.0, 4.2, 0.1)
                speed = st.slider("Tablet Speed", 50.0, 150.0, 100.0, 5.0)
                hardness = st.slider("Tablet Hardness", 50.0, 200.0, 120.0, 5.0)
                stiffness = st.slider("Stiffness", 0.5, 3.0, 1.5, 0.1)
                
                # Additional parameters
                st.markdown("**Additional Parameters:**")
                batch_size = st.slider("Batch Size", 1000, 10000, 5000, 100)
                strength = st.slider("Tablet Strength", 0.5, 2.0, 1.0, 0.1)
                size = st.slider("Tablet Size", 0.5, 2.0, 1.0, 0.1)
            
            with col_input2:
                st.markdown("**Raw Material Properties:**")
                api_content = st.slider("API Content", 0.1, 2.0, 1.0, 0.1)
                api_impurities = st.slider("API Impurities", 0.0, 1.0, 0.1, 0.01)
                lactose_water = st.slider("Lactose Water", 0.1, 2.0, 1.0, 0.1)
                smcc_water = st.slider("SMCC Water", 0.1, 2.0, 1.0, 0.1)
                starch_water = st.slider("Starch Water", 0.1, 2.0, 1.0, 0.1)
            
            # Make ML predictions
            st.subheader("ML Model Predictions")
            
            # Prepare parameters
            parameters = {
                'api_water': api_water,
                'main_CompForce mean': compression,
                'tbl_speed_mean': speed,
                'stiffness_mean': stiffness,
                'Batch Size (tablets)': batch_size,
                'strength': strength,
                'size': size,
                'api_content': api_content,
                'api_total_impurities': api_impurities,
                'lactose_water': lactose_water,
                'smcc_water': smcc_water,
                'starch_water': starch_water
            }
            
            # Get predictions from ML models
            predictions = ml_predictor.predict_all(parameters)
            
            if predictions:
                col_pred1, col_pred2 = st.columns(2)
                
                with col_pred1:
                    st.markdown("**Quality Predictions:**")
                
                    # Dissolution predictions - REMOVED BOXES
                    if 'dissolution_av' in predictions:
                        dissolution = predictions['dissolution_av']
                        if dissolution >= 90:
                            risk_class = "Low"
                            risk_color = "🟢"
                        elif dissolution >= 80:
                            risk_class = "Marginal"
                            risk_color = "🟡"
                        else:
                            risk_class = "Fail"
                            risk_color = "🔴"
                        
                        st.write(f"**Dissolution Av:** {dissolution:.2f}% {risk_color}")
                        st.write(f"Risk Level: {risk_class}")
                    
                    if 'dissolution_min' in predictions:
                        dissolution_min = predictions['dissolution_min']
                        if dissolution_min >= 85:
                            risk_class = "Low"
                            risk_color = "🟢"
                        elif dissolution_min >= 75:
                            risk_class = "Marginal"
                            risk_color = "🟡"
                        else:
                            risk_class = "Fail"
                            risk_color = "🔴"
                        
                        st.write(f"**Dissolution Min:** {dissolution_min:.2f}% {risk_color}")
                        st.write(f"Risk Level: {risk_class}")
                    
                    if 'batch_yield' in predictions:
                        yield_pred = predictions['batch_yield']
                        if yield_pred >= 95:
                            risk_class = "High"
                            risk_color = "🟢"
                        elif yield_pred >= 90:
                            risk_class = "Medium"
                            risk_color = "🟡"
                        else:
                            risk_class = "Low"
                            risk_color = "🔴"
                        
                        st.write(f"**Batch Yield:** {yield_pred:.2f}% {risk_color}")
                        st.write(f"Risk Level: {risk_class}")
                
                with col_pred2:
                    st.markdown("**Impurity Predictions:**")
                    
                    # Impurity predictions - REMOVED BOXES
                    if 'impurities_total' in predictions:
                        impurities = predictions['impurities_total']
                        st.write(f"**Impurities Total:** {impurities:.4f}")
                    
                    if 'impurity_o' in predictions:
                        impurity_o = predictions['impurity_o']
                        st.write(f"**Impurity O:** {impurity_o:.4f}")
                    
                    if 'impurity_l' in predictions:
                        impurity_l = predictions['impurity_l']
                        st.write(f"**Impurity L:** {impurity_l:.4f}")
                    
                    if 'resodual_solvent' in predictions:
                        solvent = predictions['resodual_solvent']
                        st.write(f"**Residual Solvent:** {solvent:.4f}")
                
                # Risk assessment
                st.subheader("Risk Assessment")
                
                # Calculate overall risk
                high_risk_factors = []
                if 'dissolution_av' in predictions and predictions['dissolution_av'] < 80:
                    high_risk_factors.append("Dissolution < 80%")
                if 'batch_yield' in predictions and predictions['batch_yield'] < 90:
                    high_risk_factors.append("Yield < 90%")
                if 'impurities_total' in predictions and predictions['impurities_total'] > 0.5:
                    high_risk_factors.append("High impurities")
                
                if high_risk_factors:
                    risk_level = "HIGH RISK"
                    risk_class = "risk-high"
                    risk_message = f"HIGH RISK: {'; '.join(high_risk_factors)} - immediate attention required"
                elif 'dissolution_av' in predictions and 80 <= predictions['dissolution_av'] < 90:
                    risk_level = "MEDIUM RISK"
                    risk_class = "risk-medium"
                    risk_message = "MEDIUM RISK: Dissolution 80-90% - monitor closely"
                else:
                    risk_level = "LOW RISK"
                    risk_class = "risk-low"
                    risk_message = "LOW RISK: All parameters within acceptable ranges"
                
                st.write(f"**{risk_level}**: {risk_message}")
                
                # Model performance info
                st.subheader("Model Performance")
                if models:
                    best_model = max(models.items(), key=lambda x: x[1]['r2_score'])
                    best_model_name, best_model_info = best_model
                    st.info(f"Best performing model: {best_model_name} (R² = {best_model_info['r2_score']:.3f})")
    
    # Tab 3: GenAI Assistant (unchanged)
    with tab3:
        st.header("GenAI Manufacturing Assistant")
        st.info("💬 **What this tab does:** Chat with an AI assistant that can answer questions about your manufacturing data, make predictions, and provide insights using natural language. Ask about trends, compare batches, or get recommendations.")
        
        if genai_agent is None:
            st.error("GenAI agent not available. Please check your OpenAI API key and model configuration.")
        else:
            # Chat interface
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask me about your manufacturing data, predictions, or process optimization..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = genai_agent.query(prompt)
                    st.markdown(response)
                
                # Add AI response
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Example queries
            st.subheader("Example Queries")
            example_queries = [
                "What is the average dissolution rate in our dataset?",
                "Compare batch 5 vs batch 25 quality metrics",
                "What process parameters most affect dissolution?",
                "Predict dissolution for api_water=1.5, compression=4.2, speed=100",
                "How can we improve batch yield while maintaining quality?",
                "Which process parameters are most important for batch yield?"
            ]
            
            for query in example_queries:
                if st.button(f"💡 {query}", key=f"example_{query}"):
                    st.session_state.messages.append({"role": "user", "content": query})
                    with st.chat_message("user"):
                        st.markdown(query)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = genai_agent.query(query)
                        st.markdown(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
    
    # Tab 4: Experiment Simulator (unchanged)
    with tab4:
        st.header("Future Experiment Simulator")
        st.info("🔬 **What this tab does:** Simulate multiple experimental scenarios with different parameter combinations to predict outcomes before running actual experiments. Generate parameter ranges and see which combinations give the best predicted results.")
        
        col_sim1, col_sim2 = st.columns([1, 2])
        
        with col_sim1:
            st.subheader("Simulation Parameters")
            
            # Parameter ranges
            st.markdown("**API Water Range:**")
            api_water_min = st.number_input("Min API Water", 0.5, 3.0, 1.0, 0.1, key="api_min")
            api_water_max = st.number_input("Max API Water", 0.5, 3.0, 2.0, 0.1, key="api_max")
            
            st.markdown("**Compression Force Range:**")
            compression_min = st.number_input("Min Compression", 2.0, 15.0, 3.0, 0.1, key="comp_min")
            compression_max = st.number_input("Max Compression", 2.0, 15.0, 8.0, 0.1, key="comp_max")
            
            st.markdown("**Speed Range:**")
            speed_min = st.number_input("Min Speed", 50.0, 150.0, 80.0, 5.0, key="speed_min")
            speed_max = st.number_input("Max Speed", 50.0, 150.0, 120.0, 5.0, key="speed_max")
            
            # Number of scenarios
            num_scenarios = st.slider("Number of Scenarios", 5, 50, 20)
            
            # Generate scenarios button
            if st.button("🔬 Generate ML-Powered Scenarios", type="primary"):
                st.session_state.generate_scenarios = True
        
        with col_sim2:
            st.subheader("ML-Powered Simulation Results")
            
            if st.session_state.get('generate_scenarios', False):
                with st.spinner("Generating scenarios using ML models..."):
                    # Generate random scenarios
                    np.random.seed(42)  # For reproducible results
                    
                    scenarios = []
                    for i in range(num_scenarios):
                        scenario = {
                            'id': i + 1,
                            'api_water': np.random.uniform(api_water_min, api_water_max),
                            'compression': np.random.uniform(compression_min, compression_max),
                            'speed': np.random.uniform(speed_min, speed_max),
                            'stiffness': np.random.uniform(0.5, 3.0),
                            'batch_size': np.random.randint(1000, 10000)
                        }
                        scenarios.append(scenario)
                    
                    # Make ML predictions for each scenario
                    for scenario in scenarios:
                        parameters = {
                            'api_water': scenario['api_water'],
                            'main_CompForce mean': scenario['compression'],
                            'tbl_speed_mean': scenario['speed'],
                            'stiffness_mean': scenario['stiffness'],
                            'Batch Size (tablets)': scenario['batch_size'],
                            'strength': 1.0,
                            'size': 1.0
                        }
                        
                        # Get ML predictions
                        predictions = ml_predictor.predict_all(parameters)
                        
                        # Store predictions
                        scenario['predicted_dissolution'] = predictions.get('dissolution_av', 85.0)
                        scenario['predicted_yield'] = predictions.get('batch_yield', 95.0)
                        scenario['predicted_impurities'] = predictions.get('impurities_total', 0.1)
                    
                    # Convert to DataFrame
                    scenarios_df = pd.DataFrame(scenarios)
                    
                    # Sort by predicted dissolution
                    scenarios_df = scenarios_df.sort_values('predicted_dissolution', ascending=False)
                    
                    # Display results
                    st.success(f"Generated {len(scenarios)} scenarios using ML models!")
                    
                    # Show top scenarios
                    st.subheader("Top Performing Scenarios (ML-Predicted)")
                    top_scenarios = scenarios_df.head(10)
                    
                    for _, scenario in top_scenarios.iterrows():
                        dissolution = scenario['predicted_dissolution']
                        yield_pred = scenario['predicted_yield']
                        impurities = scenario['predicted_impurities']
                        
                        if dissolution >= 90:
                            status = "🟢 Excellent"
                        elif dissolution >= 85:
                            status = "🟡 Good"
                        else:
                            status = "🔴 Poor"
                        
                        st.markdown(f"""
                        **Scenario {scenario['id']}** - {status}
                        - API Water: {scenario['api_water']:.2f}
                        - Compression: {scenario['compression']:.2f} kN
                        - Speed: {scenario['speed']:.0f} rpm
                        - **ML Predicted Dissolution: {dissolution:.1f}%**
                        - **ML Predicted Yield: {yield_pred:.1f}%**
                        - **ML Predicted Impurities: {impurities:.3f}%**
                        """)
                    
                    # Visualization
                    fig_scenarios = px.scatter(scenarios_df, 
                                            x='api_water', 
                                            y='predicted_dissolution',
                                            color='predicted_dissolution',
                                            size='compression',
                                            hover_data=['speed', 'stiffness', 'predicted_yield', 'predicted_impurities'],
                                            title="ML-Powered Scenario Performance: API Water vs Predicted Dissolution",
                                            labels={'api_water': 'API Water Content',
                                                   'predicted_dissolution': 'ML Predicted Dissolution (%)'})
                    st.plotly_chart(fig_scenarios, use_container_width=True)
                    
                    # Download results
                    csv = scenarios_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download ML Simulation Results",
                        data=csv,
                        file_name=f"ml_experiment_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # Reset the flag
                st.session_state.generate_scenarios = False
    
    # Tab 5: Scenario Generator (unchanged)
    with tab5:
        st.header("AI-Powered Scenario Generator")
        st.info("🎯 **What this tab does:** Let AI suggest optimal experimental parameters based on your quality targets. Set your desired outcomes (dissolution %, yield %, etc.) and get AI recommendations for process parameters that should achieve those targets.")
        
        col_target1, col_target2 = st.columns(2)
        
        with col_target1:
            st.subheader("Quality Targets")
            
            target_dissolution = st.slider("Target Dissolution (%)", 80.0, 100.0, 90.0, 1.0)
            target_yield = st.slider("Target Yield (%)", 85.0, 100.0, 95.0, 1.0)
            target_impurities = st.slider("Max Impurities (%)", 0.0, 1.0, 0.3, 0.01)
            
            # Constraints
            st.subheader("Process Constraints")
            max_compression = st.number_input("Max Compression Force (kN)", 2.0, 20.0, 10.0, 0.1)
            max_speed = st.number_input("Max Tablet Speed (rpm)", 50.0, 200.0, 150.0, 5.0)
            max_api_water = st.number_input("Max API Water (%)", 0.5, 5.0, 3.0, 0.1)
        
        with col_target2:
            st.subheader("ML-Powered AI Recommendations")
            
            if st.button("🤖 Generate ML-Powered AI Recommendations", type="primary"):
                with st.spinner("AI is using ML models to optimize parameters..."):
                    # Use ML optimization
                    targets = {
                        'dissolution_av': target_dissolution,
                        'batch_yield': target_yield,
                        'impurities_total': target_impurities
                    }
                    
                    # Define bounds for optimization
                    bounds = [
                        (0.5, max_api_water),  # api_water
                        (2.0, max_compression),  # compression
                        (50.0, max_speed),  # speed
                        (0.5, 3.0),  # stiffness
                        (1000, 10000),  # batch_size
                        (0.5, 2.0),  # strength
                        (0.5, 2.0)  # size
                    ]
                    
                    # Run ML optimization
                    optimal_params, optimal_predictions, error = ml_optimizer.optimize_for_targets(
                        targets, {}, bounds
                    )
                    
                    if optimal_params is not None:
                        st.success("ML-Powered AI Recommendations Generated!")
                        
                        # Display recommendations
                        st.markdown("### 🎯 ML-Optimized Parameters")
                        
                        rec_col1, rec_col2 = st.columns(2)
                        
                        with rec_col1:
                            st.metric("API Water Content", f"{optimal_params['api_water']:.2f}%")
                            st.metric("Compression Force", f"{optimal_params['main_CompForce mean']:.1f} kN")
                            st.metric("Tablet Speed", f"{optimal_params['tbl_speed_mean']:.0f} rpm")
                        
                        with rec_col2:
                            st.metric("Stiffness", f"{optimal_params['stiffness_mean']:.1f}")
                            st.metric("Batch Size", f"{optimal_params['Batch Size (tablets)']:.0f}")
                            st.metric("Optimization Error", f"{error:.3f}")
                        
                        # Expected outcomes
                        st.markdown("### 📊 ML-Predicted Outcomes")
                        
                        if optimal_predictions:
                            outcome_col1, outcome_col2, outcome_col3 = st.columns(3)
                            
                            with outcome_col1:
                                pred_dissolution = optimal_predictions.get('dissolution_av', 0)
                                st.metric("Predicted Dissolution", f"{pred_dissolution:.1f}%")
                                st.metric("Target", f"{target_dissolution:.1f}%")
                                st.metric("Difference", f"{pred_dissolution - target_dissolution:.1f}%")
                            
                            with outcome_col2:
                                pred_yield = optimal_predictions.get('batch_yield', 0)
                                st.metric("Predicted Yield", f"{pred_yield:.1f}%")
                                st.metric("Target", f"{target_yield:.1f}%")
                                st.metric("Difference", f"{pred_yield - target_yield:.1f}%")
                            
                            with outcome_col3:
                                pred_impurities = optimal_predictions.get('impurities_total', 0)
                                st.metric("Predicted Impurities", f"{pred_impurities:.3f}%")
                                st.metric("Target", f"{target_impurities:.3f}%")
                                st.metric("Difference", f"{pred_impurities - target_impurities:.3f}%")
                        
                        # Risk assessment
                        risk_factors = []
                        if pred_dissolution < target_dissolution - 2:
                            risk_factors.append("Dissolution below target")
                        if pred_yield < target_yield - 2:
                            risk_factors.append("Yield below target")
                        if pred_impurities > target_impurities + 0.05:
                            risk_factors.append("Impurities above target")
                        
                        if risk_factors:
                            st.warning(f"⚠️ Risk factors: {', '.join(risk_factors)}")
                        else:
                            st.success("✅ All targets achievable with ML-optimized parameters")
                        
                        # Implementation notes
                        st.markdown("### 📝 ML-Based Implementation Notes")
                        st.markdown("""
                        - **ML Models Used**: CatBoost models trained on historical data
                        - **Optimization Method**: Differential Evolution (global optimization)
                        - **Parameter Confidence**: Based on model R² scores and prediction accuracy
                        - **Validation**: These parameters were optimized using your trained ML models
                        
                        **Next Steps:**
                        1. Validate ML-optimized parameters in small-scale trials
                        2. Monitor key quality attributes during production
                        3. Adjust parameters based on real-time feedback
                        4. Document results for future ML model improvement
                        """)
                    else:
                        st.error("ML optimization failed. Please try different target values or constraints.")
    
    # Tab 6: ML Explorer - FIXED: Remove last plot
    with tab6:
        st.header("ML Model Performance Explorer")
        st.info("📈 **What this tab does:** Analyze the performance of your trained ML models, view feature importance, compare model accuracy, and get recommendations on which models to use for different predictions.")
        
        if not models:
            st.warning("No trained models found. Please run Phase 2 first.")
        else:
            # Model performance overview
            st.subheader("Model Performance Overview")
            
            # Convert models to DataFrame for display
            model_data = []
            for model_name, model_info in models.items():
                model_data.append({
                    'Model': model_name,
                    'Target': model_info['target'],
                    'Algorithm': model_info['model_type'],
                    'R² Score': model_info['r2_score'],
                    'RMSE': model_info['rmse'],
                    'MAE': model_info['mae']
                })
            
            model_df = pd.DataFrame(model_data)
            
            # Display model performance table
            st.dataframe(model_df, use_container_width=True)
            
            # Performance visualization
            col_perf1, col_perf2 = st.columns(2)
            
            with col_perf1:
                # R² Score comparison
                fig_r2 = px.bar(model_df, x='Model', y='R² Score', 
                              title="Model R² Score Comparison",
                              color='R² Score',
                              color_continuous_scale='viridis')
                fig_r2.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_r2, use_container_width=True)
            
            with col_perf2:
                # RMSE comparison
                fig_rmse = px.bar(model_df, x='Model', y='RMSE',
                                title="Model RMSE Comparison",
                                color='RMSE',
                                color_continuous_scale='plasma')
                fig_rmse.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_rmse, use_container_width=True)
            
            # Feature importance analysis
            st.subheader("Feature Importance Analysis")
            
            if importance_data:
                # Select target for feature importance
                target_options = list(importance_data.keys())
                selected_target = st.selectbox("Select Target Variable", target_options)
                
                if selected_target:
                    importance_df = importance_data[selected_target]
                    
                    # Display top features
                    top_features = importance_df.head(15)
                    
                    fig_importance = px.bar(top_features, 
                                          x='importance', 
                                          y='feature',
                                          orientation='h',
                                          title=f"Top 15 Most Important Features for {selected_target}",
                                          labels={'importance': 'Feature Importance',
                                                 'feature': 'Feature Name'})
                    fig_importance.update_layout(height=500)
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Feature importance table
                    st.markdown("**Detailed Feature Importance:**")
                    st.dataframe(importance_df, use_container_width=True)
            
            # Model recommendations
            st.subheader("Model Recommendations")
            
            # Group by target and find best model for each
            best_models = model_df.groupby('Target').apply(lambda x: x.loc[x['R² Score'].idxmax()])
            
            for target, best_model in best_models.iterrows():
                st.markdown(f"""
                **{target.replace('_', ' ').title()}**: 
                - Best Model: {best_model['Model']}
                - R² Score: {best_model['R² Score']:.3f}
                - RMSE: {best_model['RMSE']:.3f}
                """)
            
            # Model comparison
            st.subheader("Model Comparison")
            
            # Compare algorithms
            algorithm_comparison = model_df.groupby('Algorithm').agg({
                'R² Score': 'mean',
                'RMSE': 'mean',
                'MAE': 'mean'
            }).round(3)
            
            st.markdown("**Average Performance by Algorithm:**")
            st.dataframe(algorithm_comparison, use_container_width=True)
            

if __name__ == "__main__":
    main() 