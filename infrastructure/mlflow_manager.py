#!/usr/bin/env python3
"""
Simple MLflow Manager for Pipeline Integration
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

class MLflowManager:
    """Simple MLflow manager for model registry operations"""
    
    def __init__(self, model_prefix: str = "Pipeline"):
        self.tracking_uri = "http://mlflow.mlflow.svc.cluster.local:5000"
        self.cpu_model_name = f"CPU_Prophet_Model_{model_prefix}"
        self.memory_model_name = f"Memory_Prophet_Model_{model_prefix}"
        
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        print(f"Connected to MLflow at {self.tracking_uri}")
    
    def save_models(self, cpu_model, memory_model, stage: str = "Staging", metrics: Dict[str, Any] = None) -> Tuple[str, str]:
        """Save both models to MLflow Registry"""
        
        if metrics is None:
            metrics = {}
        
        cpu_run_id = None
        memory_run_id = None
        
        # Save CPU Model
        with mlflow.start_run(run_name=f"CPU_Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log parameters
            mlflow.log_params({
                "model_type": "prophet",
                "target_metric": "cpu_usage",
                "daily_seasonality": True,
                "weekly_seasonality": True,
                "yearly_seasonality": False
            })
            
            # Log custom metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Register model
            mlflow.sklearn.log_model(
                sk_model=cpu_model,
                artifact_path="model",
                registered_model_name=self.cpu_model_name
            )
            
            cpu_run_id = mlflow.active_run().info.run_id
            print(f"CPU Model logged as {self.cpu_model_name}")
        
        # Save Memory Model
        with mlflow.start_run(run_name=f"Memory_Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log parameters
            mlflow.log_params({
                "model_type": "prophet",
                "target_metric": "memory_usage",
                "daily_seasonality": True,
                "weekly_seasonality": True,
                "yearly_seasonality": False
            })
            
            # Log custom metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Register model
            mlflow.sklearn.log_model(
                sk_model=memory_model,
                artifact_path="model",
                registered_model_name=self.memory_model_name
            )
            
            memory_run_id = mlflow.active_run().info.run_id
            print(f"Memory Model logged as {self.memory_model_name}")
        
        # Promote models to staging
        self._promote_models_to_stage(stage)
        
        return cpu_run_id, memory_run_id
    
    def _promote_models_to_stage(self, stage: str):
        """Promote both models to specified stage"""
        for model_name in [self.cpu_model_name, self.memory_model_name]:
            try:
                latest_versions = self.client.get_latest_versions(model_name, stages=["None"])
                if latest_versions:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=latest_versions[0].version,
                        stage=stage,
                        archive_existing_versions=True
                    )
                    print(f"{model_name} promoted to {stage}")
            except Exception as e:
                print(f"Failed to promote {model_name}: {e}")
