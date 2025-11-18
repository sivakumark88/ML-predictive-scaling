#!/usr/bin/env python3
"""
MLOps Component: Model Validation Component (MLflow VERSION)
This component loads models from MLflow Model Registry for validation
"""

from kfp.dsl import component

# Using pre-built image (will need mlflow installed)
DOCKER_IMAGE = "shivapondicherry/forecast-train:latest"

@component(base_image=DOCKER_IMAGE)
def model_validation_component() -> dict:
    """
    Stage 4: Model Validation Component with MLflow
    Loads models from MLflow Model Registry instead of MinIO
    """
    import sys
    import os
    import json
    
    print("=== STAGE 4: MODEL VALIDATION (MLflow APPROACH) ===")
    
    try:
        # Add paths to import ML Engineer's scripts and MLflow manager
        sys.path.append('/opt/mlpipeline/launch')
        sys.path.append('.')
        sys.path.append('./scripts')
        sys.path.append('../infrastructure')  # For mlflow_manager.py
        
        # Import ML Engineer's ModelValidator class and MLflow
        from model_validator import ModelValidator
        from mlflow_manager import MLflowManager
        
        print("✅ Successfully imported ML Engineer's ModelValidator and MLflowManager")
        
        # Initialize MLflow (replaces MinIO initialization)
        mlflow_manager = MLflowManager()
        
        # Use ML Engineer's validation logic
        validator = ModelValidator(test_periods=24)
        
        # Load models from MLflow Registry (replaces MinIO download)
        print("📥 Loading models from MLflow Model Registry")
        cpu_model, memory_model = mlflow_manager.load_cpu_memory_models(stage="Staging")
        
        if cpu_model is None or memory_model is None:
            print("❌ Failed to load models from MLflow Registry")
            return {
                'validation_report': {'overall_status': 'FAILED'},
                'deployment_ready': False,
                'error': 'Models not found in MLflow Registry'
            }
        
        print("✅ Models loaded from MLflow Registry successfully")
        
        # Create temporary pickle files for validator compatibility
        import pickle
        with open("cpu_prophet_model.pkl", "wb") as f:
            pickle.dump(cpu_model, f)
        with open("memory_prophet_model.pkl", "wb") as f:
            pickle.dump(memory_model, f)
        
        # Validate models using ML Engineer's logic
        validation_report = validator.load_and_validate_models("cpu_prophet_model.pkl", "memory_prophet_model.pkl")
        
        # Create deployment readiness report
        readiness_report = validator.create_deployment_readiness_report(validation_report)
        
        # If validation passed, promote models to Production
        if readiness_report['deployment_ready']:
            print("🚀 Models passed validation - promoting to Production")
            mlflow_manager.promote_model(mlflow_manager.cpu_model_name, "Production")
            mlflow_manager.promote_model(mlflow_manager.memory_model_name, "Production")
            print("✅ Models promoted to Production stage in MLflow")
        
        print(f"✅ ML Engineer's model validation complete: {validation_report['overall_status']}")
        print(f"🚀 Deployment ready: {readiness_report['deployment_ready']}")
        
        return {
            'validation_report': validation_report,
            'deployment_ready': readiness_report['deployment_ready'],
            'model_storage': 'mlflow_registry',
            'models_promoted': readiness_report['deployment_ready'],
            'mlflow_models': {
                'cpu': mlflow_manager.cpu_model_name,
                'memory': mlflow_manager.memory_model_name
            }
        }
        
    except ImportError as e:
        print(f"⚠️  Could not import ML Engineer's script or MLflow: {e}")
        print("📦 Using fallback model validation...")
        
        # Try to import MLflow manager for fallback
        try:
            from mlflow_manager import MLflowManager
            mlflow_manager = MLflowManager()
            use_mlflow = True
            print("✅ MLflow manager available for fallback")
        except ImportError:
            print("⚠️  MLflow manager not available, using MinIO fallback")
            use_mlflow = False
        
        if use_mlflow:
            # MLflow fallback
            print("📥 Loading models from MLflow Registry (fallback)")
            cpu_model, memory_model = mlflow_manager.load_cpu_memory_models(stage="Staging")
            
            if cpu_model is None or memory_model is None:
                print("❌ Failed to load models from MLflow Registry")
                return {
                    'validation_report': {'overall_status': 'FAILED'},
                    'deployment_ready': False,
                    'error': 'Models not found in MLflow Registry'
                }
                
            # Direct validation using loaded models
            print("🔍 Running validation tests")
            
            future_cpu = cpu_model.make_future_dataframe(periods=24, freq='H')
            future_memory = memory_model.make_future_dataframe(periods=24, freq='H')
            
            cpu_forecast = cpu_model.predict(future_cpu)
            memory_forecast = memory_model.predict(future_memory)
            
            # Validation checks
            cpu_valid = len(cpu_forecast) > 0 and not cpu_forecast['yhat'].isnull().any()
            memory_valid = len(memory_forecast) > 0 and not memory_forecast['yhat'].isnull().any()
            reasonable_cpu = (cpu_forecast['yhat'].min() >= 0) and (cpu_forecast['yhat'].max() <= 200)
            reasonable_memory = (memory_forecast['yhat'].min() >= 0) and (memory_forecast['yhat'].max() <= 200)
            
            overall_valid = cpu_valid and memory_valid and reasonable_cpu and reasonable_memory
            
            validation_status = "PASSED" if overall_valid else "FAILED"
            
            # If validation passed, promote models to Production
            if overall_valid:
                print("🚀 Models passed validation - promoting to Production")
                mlflow_manager.promote_model(mlflow_manager.cpu_model_name, "Production")
                mlflow_manager.promote_model(mlflow_manager.memory_model_name, "Production")
                print("✅ Models promoted to Production stage in MLflow")
            
            return {
                'validation_report': {
                    'overall_status': validation_status,
                    'cpu_model': {
                        'validation_passed': cpu_valid and reasonable_cpu,
                        'prediction_count': len(cpu_forecast),
                        'min_prediction': float(cpu_forecast['yhat'].min()),
                        'max_prediction': float(cpu_forecast['yhat'].max())
                    },
                    'memory_model': {
                        'validation_passed': memory_valid and reasonable_memory,
                        'prediction_count': len(memory_forecast),
                        'min_prediction': float(memory_forecast['yhat'].min()),
                        'max_prediction': float(memory_forecast['yhat'].max())
                    }
                },
                'deployment_ready': overall_valid,
                'model_storage': 'mlflow_registry_fallback',
                'models_promoted': overall_valid
            }
        else:
            # MinIO fallback: Direct validation logic
            from minio import Minio
            import pandas as pd
            import pickle
            from prophet import Prophet
            import warnings
            warnings.filterwarnings('ignore')
            
            minio_client = Minio(
                "minio-service.kubeflow.svc.cluster.local:9000",
                access_key="minio",
                secret_key="minio123", 
                secure=False
            )
            
            print("📥 Downloading MODULAR models for validation")
            minio_client.fget_object("mlpipeline", "models/modular-cpu_prophet_model.pkl", "cpu_prophet_model.pkl")
            minio_client.fget_object("mlpipeline", "models/modular-memory_prophet_model.pkl", "memory_prophet_model.pkl")
            
            # Load models
            print("📂 Loading models for validation")
            with open("cpu_prophet_model.pkl", "rb") as f:
                cpu_model = pickle.load(f)
            
            with open("memory_prophet_model.pkl", "rb") as f:
                memory_model = pickle.load(f)
            
            # Validation logic (from ML Engineer's script)
            print("🔍 Running validation tests")
            
            future_cpu = cpu_model.make_future_dataframe(periods=24, freq='H')
            future_memory = memory_model.make_future_dataframe(periods=24, freq='H')
            
            cpu_forecast = cpu_model.predict(future_cpu)
            memory_forecast = memory_model.predict(future_memory)
            
            # Validation checks
            cpu_valid = len(cpu_forecast) > 0 and not cpu_forecast['yhat'].isnull().any()
            memory_valid = len(memory_forecast) > 0 and not memory_forecast['yhat'].isnull().any()
            reasonable_cpu = (cpu_forecast['yhat'].min() >= 0) and (cpu_forecast['yhat'].max() <= 200)
            reasonable_memory = (memory_forecast['yhat'].min() >= 0) and (memory_forecast['yhat'].max() <= 200)
            
            overall_valid = cpu_valid and memory_valid and reasonable_cpu and reasonable_memory
            
            validation_status = "PASSED" if overall_valid else "FAILED"
            
            print(f"✅ CPU Model: {'VALID' if cpu_valid else 'INVALID'}")
            print(f"🧠 Memory Model: {'VALID' if memory_valid else 'INVALID'}")
            print(f"📊 Overall validation: {validation_status}")
            
            return {
                'validation_report': {
                    'overall_status': validation_status,
                    'cpu_model': {
                        'validation_passed': cpu_valid and reasonable_cpu,
                        'prediction_count': len(cpu_forecast),
                        'min_prediction': float(cpu_forecast['yhat'].min()),
                        'max_prediction': float(cpu_forecast['yhat'].max())
                    },
                    'memory_model': {
                        'validation_passed': memory_valid and reasonable_memory,
                        'prediction_count': len(memory_forecast),
                        'min_prediction': float(memory_forecast['yhat'].min()),
                        'max_prediction': float(memory_forecast['yhat'].max())
                    }
                },
                'deployment_ready': overall_valid,
                'model_storage': 'minio_fallback',
                'model_paths_validated': {
                    'cpu': 'models/modular-cpu_prophet_model.pkl',
                    'memory': 'models/modular-memory_prophet_model.pkl'
                }
            }

if __name__ == "__main__":
    print("Model Validation Component - MLflow version with fallback support")
