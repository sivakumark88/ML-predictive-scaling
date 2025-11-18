#!/usr/bin/env python3
"""
Simple Model Training Script
Trains models and uploads ONLY to MLflow (no fallback)
"""

import pandas as pd
from minio import Minio
from prophet import Prophet
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.tracking import MlflowClient
from datetime import datetime
from typing import Tuple

class ModelTrainer:
    """Simple model trainer with MLflow-only storage"""
    
    def __init__(self, model_prefix: str = "Pipeline"):
        self.model_prefix = model_prefix
        
        # MLflow setup
        self.tracking_uri = "http://mlflow.mlflow.svc.cluster.local:5000"
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        # Model names
        self.cpu_model_name = f"CPU_Prophet_Model_{model_prefix}"
        self.memory_model_name = f"Memory_Prophet_Model_{model_prefix}"
        
        # MinIO for data loading only
        self.minio_client = Minio(
            "minio-service.kubeflow.svc.cluster.local:9000",
            access_key="minio", 
            secret_key="minio123", 
            secure=False
        )
        
        print(f"Connected to MLflow at {self.tracking_uri}")
        
    def train_models(self) -> Tuple[str, str]:
        """Train CPU and Memory models"""
        print("Loading prophet_ready_data.csv for training...")
        
        # Load processed data from MinIO
        self.minio_client.fget_object("mlpipeline", "prophet_ready_data.csv", "prophet_ready_data.csv")
        df = pd.read_csv("prophet_ready_data.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"Training with {len(df)} records")
        
        # Train CPU model
        print("Training CPU Prophet model...")
        cpu_data = df[['timestamp', 'cpu_mean_5m']].rename(columns={
            'timestamp': 'ds',
            'cpu_mean_5m': 'y'
        })
        
        cpu_model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        cpu_model.fit(cpu_data)
        print("CPU model training complete")
        
        # Train Memory model
        print("Training Memory Prophet model...")
        memory_data = df[['timestamp', 'mem_mean_5m']].rename(columns={
            'timestamp': 'ds',
            'mem_mean_5m': 'y'
        })
        
        memory_model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        memory_model.fit(memory_data)
        print("Memory model training complete")
        
        # Save to MLflow ONLY
        cpu_run_id = self._save_cpu_model(cpu_model)
        memory_run_id = self._save_memory_model(memory_model)
        
        return cpu_run_id, memory_run_id
    
    def _save_cpu_model(self, model: Prophet) -> str:
        """Save CPU model to MLflow Registry"""
        with mlflow.start_run(run_name=f"CPU_Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log parameters
            mlflow.log_params({
                "model_type": "prophet",
                "target": "cpu_usage",
                "daily_seasonality": True,
                "weekly_seasonality": True,
                "yearly_seasonality": False
            })
            
            # Save Prophet model using pickle-based approach with MLflow
            import pickle
            import os
            
            # Save model as pickle file first
            model_path = "/tmp/cpu_prophet_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Log the pickle file as artifact and register
            mlflow.log_artifact(model_path, "model")
            
            # Create a registered model entry
            try:
                mlflow.register_model(
                    model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
                    name=self.cpu_model_name
                )
            except Exception as e:
                print(f"Model registration info: {e}")
                # Model might already exist, that's okay
            
            run_id = mlflow.active_run().info.run_id
            print(f"CPU model saved to MLflow: {self.cpu_model_name}")
            
            # Promote to Staging
            self._promote_to_staging(self.cpu_model_name)
            
            return run_id
    
    def _save_memory_model(self, model: Prophet) -> str:
        """Save Memory model to MLflow Registry"""
        with mlflow.start_run(run_name=f"Memory_Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log parameters
            mlflow.log_params({
                "model_type": "prophet",
                "target": "memory_usage",
                "daily_seasonality": True,
                "weekly_seasonality": True,
                "yearly_seasonality": False
            })
            
            # Save Prophet model using pickle-based approach with MLflow
            import pickle
            import os
            
            # Save model as pickle file first
            model_path = "/tmp/memory_prophet_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Log the pickle file as artifact and register
            mlflow.log_artifact(model_path, "model")
            
            # Create a registered model entry
            try:
                mlflow.register_model(
                    model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
                    name=self.memory_model_name
                )
            except Exception as e:
                print(f"Model registration info: {e}")
                # Model might already exist, that's okay
            
            run_id = mlflow.active_run().info.run_id
            print(f"Memory model saved to MLflow: {self.memory_model_name}")
            
            # Promote to Staging
            self._promote_to_staging(self.memory_model_name)
            
            return run_id
    
    def _promote_to_staging(self, model_name: str):
        """Promote model to Staging stage"""
        try:
            latest_versions = self.client.get_latest_versions(model_name, stages=["None"])
            if latest_versions:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=latest_versions[0].version,
                    stage="Staging",
                    archive_existing_versions=True
                )
                print(f"{model_name} promoted to Staging")
        except Exception as e:
            print(f"Failed to promote {model_name}: {e}")

def run_training():
    """Main function to run training"""
    trainer = ModelTrainer()
    return trainer.train_models()

if __name__ == "__main__":
    cpu_run, memory_run = run_training()
    print(f"Training complete - CPU: {cpu_run}, Memory: {memory_run}")
