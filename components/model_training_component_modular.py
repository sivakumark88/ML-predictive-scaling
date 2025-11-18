#!/usr/bin/env python3
"""
MLOps Component: Model Training Component (MLflow VERSION)
This component uses MLflow Model Registry instead of MinIO for model storage
"""

from kfp.dsl import component

# Using pre-built image (will need mlflow installed)
DOCKER_IMAGE = "shivapondicherry/forecast-train:latest"

@component(base_image=DOCKER_IMAGE)
def model_training_component() -> dict:
    """
    Stage 3: Model Training Component with MLflow
    Stores models in MLflow Model Registry instead of MinIO
    """
    import sys
    import os
    import json
    
    print("=== STAGE 3: MODEL TRAINING (MLflow APPROACH) ===")
    
    try:
        # Add paths to import ML Engineer's scripts and MLflow manager
        sys.path.append('/opt/mlpipeline/launch')
        sys.path.append('.')
        sys.path.append('./scripts')
        sys.path.append('../infrastructure')  # For mlflow_manager.py
        
        # Import ML Engineer's ProphetTrainer class
        from model_trainer import ProphetTrainer
        from mlflow_manager import MLflowManager
        
        print("✅ Successfully imported ML Engineer's ProphetTrainer and MLflowManager")
        
        # Initialize MLflow (replaces MinIO initialization)
        mlflow_manager = MLflowManager()
        
        # Use ML Engineer's training logic
        trainer = ProphetTrainer()
        
        # Download Prophet-ready data from previous stage (still from MinIO for data)
        from minio import Minio
        minio_client = Minio(
            "minio-service.kubeflow.svc.cluster.local:9000",
            access_key="minio",
            secret_key="minio123", 
            secure=False
        )
        
        print("📥 Downloading Prophet-ready data")
        minio_client.fget_object("mlpipeline", "modular-prophet_ready_data.csv", "prophet_ready_data.csv")
        
        import pandas as pd
        df = pd.read_csv("prophet_ready_data.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Train models using ML Engineer's logic
        cpu_model, memory_model = trainer.train_models(df)
        
        # Save models using MLflow instead of MinIO
        print("📤 Saving models to MLflow Model Registry")
        
        # Calculate basic training metrics
        metrics = {
            "training_records": len(df),
            "training_completed": 1.0,
            "model_type": "prophet_forecasting"
        }
        
        # Parameters from training
        parameters = {
            "daily_seasonality": True,
            "weekly_seasonality": True, 
            "yearly_seasonality": False,
            "data_source": "modular-prophet_ready_data.csv",
            "pipeline_stage": "modular-4stage"
        }
        
        # Save both models to MLflow Registry (replaces MinIO upload)
        cpu_run_id, memory_run_id = mlflow_manager.save_models(
            cpu_model=cpu_model,
            memory_model=memory_model,
            stage="Staging",  # Initial stage
            metrics=metrics
        )
        
        print(f"✅ Models saved to MLflow - CPU: {cpu_run_id}, Memory: {memory_run_id}")
        
        return {
            'cpu_model_trained': True,
            'memory_model_trained': True,
            'training_records': len(df),
            'model_storage': 'mlflow_registry',
            'cpu_run_id': cpu_run_id,
            'memory_run_id': memory_run_id,
            'mlflow_models': {
                'cpu': mlflow_manager.cpu_model_name,
                'memory': mlflow_manager.memory_model_name
            }
        }
        
    except ImportError as e:
        print(f"⚠️  Could not import ML Engineer's script or MLflow: {e}")
        print("📦 Using fallback model training with MLflow...")
        
        # Fallback: Direct Prophet training with MLflow
        from minio import Minio
        import pandas as pd
        import pickle
        from prophet import Prophet
        import warnings
        warnings.filterwarnings('ignore')
        
        # Try to import MLflow manager for fallback
        try:
            from mlflow_manager import MLflowManager
            mlflow_manager = MLflowManager()
            use_mlflow = True
            print("✅ MLflow manager available for fallback")
        except ImportError:
            print("⚠️  MLflow manager not available, using local pickle")
            use_mlflow = False
        
        minio_client = Minio(
            "minio-service.kubeflow.svc.cluster.local:9000",
            access_key="minio",
            secret_key="minio123", 
            secure=False
        )
        
        print("📥 Downloading Prophet-ready data")
        minio_client.fget_object("mlpipeline", "modular-prophet_ready_data.csv", "prophet_ready_data.csv")
        
        df = pd.read_csv("prophet_ready_data.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"✅ Training data: {len(df)} records")
        
        # Train CPU model (from ML Engineer's logic)
        print("🤖 Training CPU forecasting model")
        cpu_data = df[['timestamp', 'cpu_mean_5m']].rename(columns={
            'timestamp': 'ds',
            'cpu_mean_5m': 'y'
        })
        
        cpu_model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        cpu_model.fit(cpu_data)
        
        # Train Memory model
        print("🧠 Training Memory forecasting model")
        memory_data = df[['timestamp', 'mem_mean_5m']].rename(columns={
            'timestamp': 'ds', 
            'mem_mean_5m': 'y'
        })
        
        memory_model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        memory_model.fit(memory_data)
        
        # Save models - MLflow or fallback
        if use_mlflow:
            print("💾 Saving models to MLflow Model Registry (fallback)")
            
            metrics = {
                "training_records": len(df),
                "training_completed": 1.0,
                "model_type": "prophet_forecasting_fallback"
            }
            
            cpu_run_id, memory_run_id = mlflow_manager.save_models(
                cpu_model=cpu_model,
                memory_model=memory_model,
                stage="Staging",
                metrics=metrics
            )
            
            return {
                'cpu_model_trained': True,
                'memory_model_trained': True,
                'training_records': len(df),
                'model_storage': 'mlflow_registry_fallback',
                'cpu_run_id': cpu_run_id,
                'memory_run_id': memory_run_id
            }
        else:
            print("💾 Saving trained models as pickle files")
            with open("cpu_prophet_model.pkl", "wb") as f:
                pickle.dump(cpu_model, f)
                
            with open("memory_prophet_model.pkl", "wb") as f:
                pickle.dump(memory_model, f)
            
            # Upload to MinIO as backup
            print("📤 Uploading models to MinIO (fallback)")
            minio_client.fput_object("mlpipeline", "models/modular-cpu_prophet_model.pkl", "cpu_prophet_model.pkl")
            minio_client.fput_object("mlpipeline", "models/modular-memory_prophet_model.pkl", "memory_prophet_model.pkl")
            
            return {
                'cpu_model_trained': True,
                'memory_model_trained': True,
                'training_records': len(df),
                'model_storage': 'minio_fallback',
                'model_paths': {
                    'cpu': 'models/modular-cpu_prophet_model.pkl',
                    'memory': 'models/modular-memory_prophet_model.pkl'
                }
            }

if __name__ == "__main__":
    print("Model Training Component - MLflow version with fallback support")
