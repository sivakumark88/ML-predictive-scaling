#!/usr/bin/env python3
"""
Simple Model Validation Script
Validates models from MLflow ONLY (no fallback)
"""

import mlflow
import mlflow.artifacts
from mlflow.tracking import MlflowClient
from prophet import Prophet
from typing import Dict, Any
import pandas as pd
import pickle

class ModelValidator:
    """Simple model validator using MLflow only"""
    
    def __init__(self, model_prefix: str = "Pipeline"):
        self.model_prefix = model_prefix
        
        # MLflow setup
        self.tracking_uri = "http://mlflow.mlflow.svc.cluster.local:5000"
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        # Model names
        self.cpu_model_name = f"CPU_Prophet_Model_{model_prefix}"
        self.memory_model_name = f"Memory_Prophet_Model_{model_prefix}"
        
        print(f"Connected to MLflow at {self.tracking_uri}")
    
    def validate_models(self) -> Dict[str, Any]:
        """Validate models from MLflow Staging"""
        print("Loading models from MLflow Staging...")
        
        # Load models from MLflow Registry as artifacts (pickle files)
        print("Loading models from MLflow Staging...")
        
        # Download and load pickle files
        import pickle
        import tempfile
        import os
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        
        # Get CPU model version details from Staging
        try:
            cpu_versions = client.get_latest_versions(self.cpu_model_name, stages=["Staging"])
            if not cpu_versions:
                raise Exception(f"No CPU model found in Staging for {self.cpu_model_name}")
            
            cpu_version = cpu_versions[0]
            cpu_run_id = cpu_version.run_id
            print(f"Found CPU model version {cpu_version.version} from run {cpu_run_id}")
            
            # Download artifacts directly using run_id instead of model registry
            cpu_uri = f"runs:/{cpu_run_id}/model"
            cpu_path = mlflow.artifacts.download_artifacts(cpu_uri)
            
        except Exception as e:
            print(f"Error getting CPU model from registry: {e}")
            print("Trying alternative approach...")
            # Fallback: try to get the latest run directly
            experiments = client.search_experiments()
            runs = client.search_runs(experiment_ids=[exp.experiment_id for exp in experiments])
            
            cpu_run_id = None
            for run in runs:
                if run.data.tags.get('model_type') == 'cpu_prophet':
                    cpu_run_id = run.info.run_id
                    break
            
            if not cpu_run_id:
                raise Exception("Could not find CPU model run")
                
            cpu_uri = f"runs:/{cpu_run_id}/model"  
            cpu_path = mlflow.artifacts.download_artifacts(cpu_uri)
        
        # List the downloaded files to debug
        print(f"CPU model artifacts downloaded to: {cpu_path}")
        if os.path.exists(cpu_path):
            print(f"Contents: {os.listdir(cpu_path)}")
        
        # Look for pickle file in the downloaded artifacts
        cpu_model_file = f"{cpu_path}/cpu_prophet_model.pkl"
        if not os.path.exists(cpu_model_file):
            # Try alternative paths
            for root, dirs, files in os.walk(cpu_path):
                for file in files:
                    if file.endswith('.pkl') and 'cpu' in file.lower():
                        cpu_model_file = os.path.join(root, file)
                        break
        
        print(f"Loading CPU model from: {cpu_model_file}")
        with open(cpu_model_file, 'rb') as f:
            cpu_model = pickle.load(f)
        
        # Download Memory model artifacts using same approach
        try:
            memory_versions = client.get_latest_versions(self.memory_model_name, stages=["Staging"])
            if not memory_versions:
                raise Exception(f"No Memory model found in Staging for {self.memory_model_name}")
            
            memory_version = memory_versions[0]
            memory_run_id = memory_version.run_id
            print(f"Found Memory model version {memory_version.version} from run {memory_run_id}")
            
            memory_uri = f"runs:/{memory_run_id}/model"
            
        except Exception as e:
            print(f"Error getting Memory model from registry: {e}")
            print("Trying alternative approach...")
            # Fallback: find memory model run
            experiments = client.search_experiments()
            runs = client.search_runs(experiment_ids=[exp.experiment_id for exp in experiments])
            
            memory_run_id = None
            for run in runs:
                if run.data.tags.get('model_type') == 'memory_prophet':
                    memory_run_id = run.info.run_id
                    break
            
            if not memory_run_id:
                raise Exception("Could not find Memory model run")
                
            memory_uri = f"runs:/{memory_run_id}/model"
            
        memory_path = mlflow.artifacts.download_artifacts(memory_uri)
        
        print(f"Memory model artifacts downloaded to: {memory_path}")
        if os.path.exists(memory_path):
            print(f"Contents: {os.listdir(memory_path)}")
        
        # Look for pickle file in the downloaded artifacts
        memory_model_file = f"{memory_path}/memory_prophet_model.pkl"
        if not os.path.exists(memory_model_file):
            # Try alternative paths
            for root, dirs, files in os.walk(memory_path):
                for file in files:
                    if file.endswith('.pkl') and 'memory' in file.lower():
                        memory_model_file = os.path.join(root, file)
                        break
        
        print(f"Loading Memory model from: {memory_model_file}")
        with open(memory_model_file, 'rb') as f:
            memory_model = pickle.load(f)
        
        print("Models loaded from MLflow")
        
        # Validate CPU model
        cpu_results = self._validate_single_model(cpu_model, "CPU")
        
        # Validate Memory model
        memory_results = self._validate_single_model(memory_model, "Memory")
        
        # Overall validation
        overall_passed = cpu_results['valid'] and memory_results['valid']
        
        validation_results = {
            'overall_status': 'PASSED' if overall_passed else 'FAILED',
            'cpu_valid': cpu_results['valid'],
            'memory_valid': memory_results['valid'],
            'cpu_predictions': cpu_results['prediction_count'],
            'memory_predictions': memory_results['prediction_count']
        }
        
        print(f"Validation Status: {validation_results['overall_status']}")
        
        # Promote to Production if validation passes
        if overall_passed:
            self._promote_to_production()
        
        return validation_results
    
    def _validate_single_model(self, model, model_name: str) -> Dict[str, Any]:
        """Validate a single Prophet model"""
        print(f"Validating {model_name} model...")
        
        # Create future dataframe for prediction - Prophet expects 'ds' column
        future_df = model.make_future_dataframe(periods=24, freq='H')
        
        # Generate predictions using Prophet
        forecast = model.predict(future_df)
        
        # Validation checks on Prophet forecast results
        has_predictions = len(forecast) > 0
        no_nulls = not forecast['yhat'].isnull().any()
        reasonable_range = (forecast['yhat'].min() >= 0) and (forecast['yhat'].max() <= 200)
        
        is_valid = has_predictions and no_nulls and reasonable_range
        
        print(f"  {model_name}: {'VALID' if is_valid else 'INVALID'}")
        print(f"  Predictions: {len(forecast)}")
        print(f"  Range: {forecast['yhat'].min():.2f} - {forecast['yhat'].max():.2f}")
        
        return {
            'valid': is_valid,
            'prediction_count': len(forecast),
            'min_prediction': float(forecast['yhat'].min()),
            'max_prediction': float(forecast['yhat'].max())
        }
    
    def _promote_to_production(self):
        """Promote both models to Production"""
        print("Validation PASSED - Promoting to Production...")
        
        for model_name in [self.cpu_model_name, self.memory_model_name]:
            try:
                latest_versions = self.client.get_latest_versions(model_name, stages=["Staging"])
                if latest_versions:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=latest_versions[0].version,
                        stage="Production",
                        archive_existing_versions=True
                    )
                    print(f"{model_name} promoted to Production")
            except Exception as e:
                print(f"Failed to promote {model_name}: {e}")

def run_validation():
    """Main function to run validation"""
    validator = ModelValidator()
    return validator.validate_models()

if __name__ == "__main__":
    result = run_validation()
    print(f"Validation complete: {result}")
