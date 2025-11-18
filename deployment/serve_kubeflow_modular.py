#!/usr/bin/env python3
"""
FastAPI Serve Script for MLflow Models
Uses models from MLflow Model Registry instead of MinIO pickle files
"""

from fastapi import FastAPI, HTTPException, Query
from datetime import datetime, timedelta
import pickle
import os
import sys
import pandas as pd
from typing import Dict, Optional
import uvicorn
from minio import Minio

# Add infrastructure path for MLflow manager
sys.path.append('/opt/mlpipeline/launch')
sys.path.append('../infrastructure')
sys.path.append('./infrastructure')

try:
    from mlflow_manager import MLflowManager
except ImportError:
    print("⚠️  MLflow Manager not available - will use MinIO fallback")
    MLflowManager = None

app = FastAPI(
    title="K8s Resource Forecasting API - MLflow Edition",
    description="""
    Resource Forecasting API for Kubernetes workloads using MLflow Model Registry.
    
    🔧 Uses MLflow Model Registry:
    - CPU_Prophet_Model (from Production or Staging stage)
    - Memory_Prophet_Model (from Production or Staging stage)
    
    Available endpoints:
    - /?datetime=YYYY-MM-DDTHH:MM:SS : Get prediction for specific datetime
    - /next_day : Get prediction for tomorrow with daily pattern analysis
    - /next_week : Get prediction with weekly pattern analysis
    - /next_weekend : Get prediction for weekend load patterns
    - /health : Health check endpoint
    - /models/info : Get information about loaded models
    """,
    version="2.0.0-mlflow",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for models
cpu_model = None
memory_model = None
model_source = "none"
mlflow_manager = None

def load_models_from_mlflow():
    """Load models from MLflow Model Registry"""
    global cpu_model, memory_model, model_source, mlflow_manager
    
    try:
        print("🔄 Loading models from MLflow Model Registry...")
        
        # Import MLflow manager
        try:
            from mlflow_manager import MLflowManager
            mlflow_manager = MLflowManager()
            print("✅ MLflow Manager initialized")
        except ImportError:
            print("❌ MLflow Manager not available, trying fallback...")
            return load_models_from_minio_fallback()
        
        # Try Production stage first, then Staging
        print("📥 Attempting to load from Production stage...")
        cpu_model, memory_model = mlflow_manager.load_cpu_memory_models(stage="Production")
        
        if cpu_model is None or memory_model is None:
            print("⚠️  No Production models found, trying Staging...")
            cpu_model, memory_model = mlflow_manager.load_cpu_memory_models(stage="Staging")
        
        if cpu_model is not None and memory_model is not None:
            model_source = "mlflow_registry"
            stage = "Production" if mlflow_manager.load_model(mlflow_manager.cpu_model_name, "Production") else "Staging"
            print(f"✅ Models loaded successfully from MLflow Registry ({stage} stage)")
            print(f"   • CPU Model: {mlflow_manager.cpu_model_name}")
            print(f"   • Memory Model: {mlflow_manager.memory_model_name}")
            return True
        else:
            print("❌ No models found in MLflow Registry, trying MinIO fallback...")
            return load_models_from_minio_fallback()
            
    except Exception as e:
        print(f"❌ Error loading from MLflow: {e}")
        print("🔄 Falling back to MinIO...")
        return load_models_from_minio_fallback()

def load_models_from_minio_fallback():
    """Fallback: Load models from MinIO (original method)"""
    global cpu_model, memory_model, model_source
    
    try:
        print("🔧 Fallback: Loading models from MinIO...")
        
        from minio import Minio
        
        # MinIO connection
        minio_client = Minio(
            "minio-service.kubeflow.svc.cluster.local:9000",
            access_key="minio",
            secret_key="minio123",
            secure=False
        )
        
        # Create models directory if it doesn't exist
        os.makedirs("/mnt/models", exist_ok=True)
        
        # Try modular models first, then regular models
        try:
            print("📥 Trying modular models...")
            minio_client.fget_object("mlpipeline", "models/modular-cpu_prophet_model.pkl", "/mnt/models/cpu_prophet_model.pkl")
            minio_client.fget_object("mlpipeline", "models/modular-memory_prophet_model.pkl", "/mnt/models/memory_prophet_model.pkl")
        except:
            print("📥 Trying regular models...")
            minio_client.fget_object("mlpipeline", "models/cpu_prophet_model.pkl", "/mnt/models/cpu_prophet_model.pkl")
            minio_client.fget_object("mlpipeline", "models/memory_prophet_model.pkl", "/mnt/models/memory_prophet_model.pkl")
        
        # Load models
        print("🤖 Loading CPU model...")
        with open("/mnt/models/cpu_prophet_model.pkl", "rb") as f:
            cpu_model = pickle.load(f)
            
        print("🧠 Loading Memory model...")
        with open("/mnt/models/memory_prophet_model.pkl", "rb") as f:
            memory_model = pickle.load(f)
            
        model_source = "minio_fallback"
        print("✅ Models loaded successfully from MinIO!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading from MinIO: {e}")
        model_source = "none"
        return False

def get_next_weekend_date():
    """Get the next Saturday at 10:00 AM"""
    current = pd.Timestamp.now()
    days_until_saturday = (5 - current.weekday()) % 7  # 5 represents Saturday
    if days_until_saturday == 0 and current.hour >= 10:
        days_until_saturday = 7
    next_saturday = current + timedelta(days=days_until_saturday)
    return next_saturday.replace(hour=10, minute=0, second=0)

def generate_summary(cpu_value: float, mem_value: float, period: str) -> dict:
    """Generate enhanced recommendations like production API"""
    
    # Memory conversion logic (same as v1.4)
    base_memory_raw = mem_value
    
    if base_memory_raw > 1000000:  # If it's in bytes
        base_memory_gb = base_memory_raw / (1024 * 1024 * 1024)  # bytes to GB
    elif base_memory_raw > 100:  # If it's percentage of large base
        base_memory_gb = base_memory_raw / 100 * 8  # Convert to reasonable GB (8GB base)
    elif base_memory_raw > 10:  # If it's percentage 
        base_memory_gb = base_memory_raw / 10  # Convert percentage to GB
    else:
        base_memory_gb = base_memory_raw  # Already in GB
    
    # CPU cores conversion (percentage to cores)
    cpu_cores = cpu_value / 100.0 if cpu_value > 1 else cpu_value
    
    # Calculate Kubernetes resource recommendations
    cpu_request_millicores = int(cpu_cores * 800)  # 80% of prediction as request
    cpu_limit_millicores = int(cpu_cores * 1200)   # 120% of prediction as limit
    
    memory_request_mb = int(base_memory_gb * 800)   # 80% of prediction as request  
    memory_limit_mb = int(base_memory_gb * 1200)    # 120% of prediction as limit
    
    # Determine confidence and insights
    if cpu_cores > 0.5:  # High CPU usage
        cpu_insight = f"High CPU expected: {cpu_cores:.2f} cores - consider optimization"
        confidence = "High" if cpu_cores < 2.0 else "Medium"
    elif cpu_cores > 0.1:
        cpu_insight = f"Moderate CPU usage: {cpu_cores:.2f} cores - normal operation"
        confidence = "High"
    else:
        cpu_insight = f"Low CPU usage: {cpu_cores:.2f} cores - potential over-provisioning"
        confidence = "Medium"
        
    if base_memory_gb > 4.0:
        memory_insight = f"High memory usage: {base_memory_gb:.1f}GB - monitor for optimization"
    elif base_memory_gb > 1.0:
        memory_insight = f"Normal memory usage: {base_memory_gb:.1f}GB - stable"
    else:
        memory_insight = f"Low memory usage: {base_memory_gb:.1f}GB - potential over-provisioning"
    
    return {
        "period": period.replace("specific_datetime_", "specific datetime "),
        "summary": "Based on modular 4-stage pipeline predictions, recommended allocations:",
        "recommendations": {
            "cpu": {
                "request": f"{cpu_request_millicores}m",
                "limit": f"{cpu_limit_millicores}m"
            },
            "memory": {
                "request": f"{memory_request_mb}Mi", 
                "limit": f"{memory_limit_mb}Mi"
            }
        },
        "insights": {
            "cpu_utilization": cpu_insight,
            "memory_utilization": memory_insight,
            "confidence": confidence
        },
        "model_info": {
            "model_version": "v1.5-modular-corrected",
            "training_date": "2025-09-15", 
            "model_type": "Prophet with corrected parameters (yearly_seasonality=True)"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Load models from MLflow Registry on startup"""
    success = load_models_from_mlflow()
    if not success:
        print("⚠️  Failed to load models from MLflow/MinIO - API may not work properly")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if (cpu_model is not None and memory_model is not None) else "not_loaded"
    
    return {
        "status": "healthy",
        "version": "2.0.0-mlflow",
        "models_loaded": model_status,
        "model_source": model_source,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models/info")
async def models_info():
    """Get detailed information about loaded models"""
    if cpu_model is None or memory_model is None:
        return {
            "error": "Models not loaded",
            "model_source": model_source
        }
    
    try:
        result = {
            "model_source": model_source,
            "models_status": "loaded",
            "timestamp": datetime.now().isoformat()
        }
        
        if model_source == "mlflow_registry" and mlflow_manager:
            # Get detailed model info from MLflow Registry
            cpu_info = mlflow_manager.get_model_info(mlflow_manager.cpu_model_name, "Production")
            memory_info = mlflow_manager.get_model_info(mlflow_manager.memory_model_name, "Production")
            
            if not cpu_info:
                cpu_info = mlflow_manager.get_model_info(mlflow_manager.cpu_model_name, "Staging")
            if not memory_info:
                memory_info = mlflow_manager.get_model_info(mlflow_manager.memory_model_name, "Staging")
            
            result.update({
                "cpu_model": {
                    "name": mlflow_manager.cpu_model_name,
                    "info": cpu_info if cpu_info else "Model info not available"
                },
                "memory_model": {
                    "name": mlflow_manager.memory_model_name,
                    "info": memory_info if memory_info else "Model info not available"
                },
                "mlflow_tracking_uri": mlflow_manager.tracking_uri
            })
        else:
            # Fallback info for MinIO models
            result.update({
                "cpu_model": {
                    "name": "cpu_prophet_model.pkl",
                    "type": "Prophet time series model",
                    "source": "MinIO fallback"
                },
                "memory_model": {
                    "name": "memory_prophet_model.pkl", 
                    "type": "Prophet time series model",
                    "source": "MinIO fallback"
                }
            })
        
        return result
        
    except Exception as e:
        return {
            "error": f"Failed to get model info: {e}",
            "model_source": model_source
        }

@app.get("/")
async def predict_datetime(datetime_str: str = Query(..., description="DateTime in format YYYY-MM-DDTHH:MM:SS")):
    """Get prediction for specific datetime using MODULAR models"""
    
    if cpu_model is None or memory_model is None:
        raise HTTPException(status_code=503, detail="MODULAR models not loaded")
    
    try:
        # Parse datetime
        pred_datetime = pd.to_datetime(datetime_str)
        
        # Create future dataframe for both models
        cpu_future = pd.DataFrame({'ds': [pred_datetime]})
        memory_future = pd.DataFrame({'ds': [pred_datetime]})
        
        # Generate predictions using MODULAR models
        cpu_forecast = cpu_model.predict(cpu_future)
        memory_forecast = memory_model.predict(memory_future)
        
        cpu_prediction = float(cpu_forecast['yhat'].iloc[0])
        memory_prediction = float(memory_forecast['yhat'].iloc[0])
        
        # Generate summary
        summary = generate_summary(cpu_prediction, memory_prediction, f"specific_datetime_{datetime_str}")
        
        return {
            "datetime": datetime_str,
            "predictions": {
                "cpu_cores": round(cpu_prediction, 4),  # Match production format
                "memory_percent": round(memory_prediction, 2)
            },
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/next_day")
async def predict_next_day():
    """Get daily average resource forecast for tomorrow"""
    
    if cpu_model is None or memory_model is None:
        raise HTTPException(status_code=503, detail="MODULAR models not loaded")
    
    try:
        # Tomorrow at noon (representative of daily average)
        tomorrow = pd.Timestamp.now().floor('D') + timedelta(days=1, hours=12)
        
        # Create future dataframes
        cpu_future = pd.DataFrame({'ds': [tomorrow]})
        memory_future = pd.DataFrame({'ds': [tomorrow]})
        
        # Generate predictions
        cpu_forecast = cpu_model.predict(cpu_future)
        memory_forecast = memory_model.predict(memory_future)
        
        cpu_cores = float(cpu_forecast['yhat'].iloc[0])
        memory_bytes = float(memory_forecast['yhat'].iloc[0])
        
        # Convert CPU percentage to cores if needed
        if cpu_cores > 1:
            cpu_cores = cpu_cores / 100.0
        
        # Generate Kubernetes recommendations
        cpu_request_m = int(cpu_cores * 0.8 * 1000)
        cpu_limit_m = int(cpu_cores * 1.2 * 1000)
        memory_gb = memory_bytes / 1e9 if memory_bytes > 1000000 else memory_bytes
        memory_request_mi = int(memory_gb * 0.8 * 1024)
        memory_limit_mi = int(memory_gb * 1.2 * 1024)
        
        return {
            "forecast_type": "daily_average",
            "date": tomorrow.strftime("%Y-%m-%d"),
            "day_of_week": tomorrow.strftime("%A"),
            "predictions": {
                "cpu_cores": round(cpu_cores, 4),
                "memory_gb": round(memory_gb, 3)
            },
            "recommendations": {
                "cpu": {
                    "request": f"{cpu_request_m}m",
                    "limit": f"{cpu_limit_m}m"
                },
                "memory": {
                    "request": f"{memory_request_mi}Mi", 
                    "limit": f"{memory_limit_mi}Mi"
                }
            },
            "insights": {
                "cpu_utilization": f"Daily average: {cpu_cores:.3f} cores - {'Low utilization' if cpu_cores < 0.05 else 'Normal utilization' if cpu_cores < 0.15 else 'High utilization'}",
                "memory_utilization": f"Daily average: {memory_gb:.1f}GB - {'Low usage' if memory_gb < 0.5 else 'Normal usage' if memory_gb < 1.0 else 'High usage'}",
                "confidence": "High"
            },
            "model_info": {
                "model_version": "v1.5-modular-enhanced",
                "training_date": "2025-09-15",
                "forecast_type": "Prophet daily average"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/next_week")
async def predict_next_week():
    """Get weekly average resource forecast for next week"""
    
    if cpu_model is None or memory_model is None:
        raise HTTPException(status_code=503, detail="MODULAR models not loaded")
    
    try:
        # Next week at noon (representative of weekly average)
        next_week = pd.Timestamp.now() + timedelta(days=7)
        next_week = next_week.replace(hour=12, minute=0, second=0, microsecond=0)
        
        # Create future dataframes
        cpu_future = pd.DataFrame({'ds': [next_week]})
        memory_future = pd.DataFrame({'ds': [next_week]})
        
        # Generate predictions
        cpu_forecast = cpu_model.predict(cpu_future)
        memory_forecast = memory_model.predict(memory_future)
        
        cpu_cores = float(cpu_forecast['yhat'].iloc[0])
        memory_bytes = float(memory_forecast['yhat'].iloc[0])
        
        # Convert CPU percentage to cores if needed
        if cpu_cores > 1:
            cpu_cores = cpu_cores / 100.0
        
        # Generate Kubernetes recommendations
        cpu_request_m = int(cpu_cores * 0.8 * 1000)
        cpu_limit_m = int(cpu_cores * 1.2 * 1000)
        memory_gb = memory_bytes / 1e9 if memory_bytes > 1000000 else memory_bytes
        memory_request_mi = int(memory_gb * 0.8 * 1024)
        memory_limit_mi = int(memory_gb * 1.2 * 1024)
        
        return {
            "forecast_type": "weekly_average",
            "week_start": next_week.strftime("%Y-%m-%d"),
            "day_of_week": next_week.strftime("%A"),
            "predictions": {
                "cpu_cores": round(cpu_cores, 4),
                "memory_gb": round(memory_gb, 3)
            },
            "recommendations": {
                "cpu": {
                    "request": f"{cpu_request_m}m",
                    "limit": f"{cpu_limit_m}m"
                },
                "memory": {
                    "request": f"{memory_request_mi}Mi",
                    "limit": f"{memory_limit_mi}Mi"
                }
            },
            "insights": {
                "cpu_utilization": f"Weekly average: {cpu_cores:.3f} cores - {'Low utilization' if cpu_cores < 0.05 else 'Normal utilization' if cpu_cores < 0.15 else 'High utilization'}",
                "memory_utilization": f"Weekly average: {memory_gb:.1f}GB - {'Low usage' if memory_gb < 0.5 else 'Normal usage' if memory_gb < 1.0 else 'High usage'}",
                "confidence": "High"
            },
            "model_info": {
                "model_version": "v1.5-modular-enhanced",
                "training_date": "2025-09-15",
                "forecast_type": "Prophet weekly average"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weekly forecast error: {str(e)}")

@app.get("/next_month")
async def predict_next_month():
    """Get monthly average resource forecast using multiple sample points"""
    
    if cpu_model is None or memory_model is None:
        raise HTTPException(status_code=503, detail="MODULAR models not loaded")
    
    try:
        # Use multiple sample points across next month for better averaging
        now = pd.Timestamp.now()
        sample_dates = []
        
        # Sample 4 points: week 2, 3, 4, 5 of next month 
        for week_offset in [7, 14, 21, 28]:
            sample_date = now + timedelta(days=week_offset)
            sample_date = sample_date.replace(hour=12, minute=0, second=0, microsecond=0)
            sample_dates.append(sample_date)
        
        # Create future dataframes with multiple points
        future_df = pd.DataFrame({'ds': sample_dates})
        
        # Generate predictions for all sample points
        cpu_forecast = cpu_model.predict(future_df)
        memory_forecast = memory_model.predict(future_df)
        
        # Average the predictions for more stable monthly estimate
        cpu_cores = float(cpu_forecast['yhat'].mean())
        memory_bytes = float(memory_forecast['yhat'].mean())
        
        # Convert CPU percentage to cores if needed
        if cpu_cores > 1:
            cpu_cores = cpu_cores / 100.0
        
        # Generate Kubernetes recommendations (more conservative for monthly)
        cpu_request_m = int(cpu_cores * 0.7 * 1000)  # More conservative 70% for monthly planning
        cpu_limit_m = int(cpu_cores * 1.5 * 1000)    # Higher headroom 150% for uncertainty
        memory_gb = memory_bytes / 1e9 if memory_bytes > 1000000 else memory_bytes
        memory_request_mi = int(memory_gb * 0.7 * 1024)  # More conservative for monthly
        memory_limit_mi = int(memory_gb * 1.5 * 1024)    # Higher headroom for monthly
        
        # Calculate prediction range for confidence
        cpu_min = float(cpu_forecast['yhat'].min())
        cpu_max = float(cpu_forecast['yhat'].max())
        if cpu_min > 1: cpu_min /= 100.0
        if cpu_max > 1: cpu_max /= 100.0
        
        return {
            "forecast_type": "monthly_average",
            "month": (now + timedelta(days=15)).strftime("%Y-%m"),
            "sampling_method": "4_week_average",
            "predictions": {
                "cpu_cores": round(cpu_cores, 4),
                "memory_gb": round(memory_gb, 3),
                "cpu_range": {
                    "min": round(cpu_min, 4),
                    "max": round(cpu_max, 4)
                }
            },
            "recommendations": {
                "cpu": {
                    "request": f"{cpu_request_m}m",
                    "limit": f"{cpu_limit_m}m"
                },
                "memory": {
                    "request": f"{memory_request_mi}Mi",
                    "limit": f"{memory_limit_mi}Mi"
                }
            },
            "insights": {
                "cpu_utilization": f"Monthly average: {cpu_cores:.3f} cores (range: {cpu_min:.3f}-{cpu_max:.3f}) - {'Low utilization' if cpu_cores < 0.05 else 'Normal utilization' if cpu_cores < 0.15 else 'High utilization'}",
                "memory_utilization": f"Monthly average: {memory_gb:.1f}GB - {'Low usage' if memory_gb < 0.5 else 'Normal usage' if memory_gb < 1.0 else 'High usage'}",
                "confidence": "Medium (4-week sampling for stability)",
                "recommendation_note": "Conservative monthly planning: 70% request (safe baseline), 150% limit (handles uncertainty)"
            },
            "model_info": {
                "model_version": "v1.5-modular-enhanced",
                "training_date": "2025-09-15",
                "forecast_type": "Prophet multi-point monthly average"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monthly forecast error: {str(e)}")

@app.get("/next_weekend")
async def predict_next_weekend():
    """Get prediction for next weekend using MODULAR models"""
    
    if cpu_model is None or memory_model is None:
        raise HTTPException(status_code=503, detail="MODULAR models not loaded")
    
    try:
        # Get next Saturday 10 AM
        next_weekend = get_next_weekend_date()
        
        # Create future dataframes  
        cpu_future = pd.DataFrame({'ds': [next_weekend]})
        memory_future = pd.DataFrame({'ds': [next_weekend]})
        
        # Generate predictions
        cpu_forecast = cpu_model.predict(cpu_future)
        memory_forecast = memory_model.predict(memory_future)
        
        cpu_prediction = float(cpu_forecast['yhat'].iloc[0])
        memory_prediction = float(memory_forecast['yhat'].iloc[0])
        
        # Generate summary
        summary = generate_summary(cpu_prediction, memory_prediction, "next_weekend")
        
        return {
            "prediction_date": next_weekend.isoformat(),
            "period": "weekend",
            "cpu_percent": round(cpu_prediction, 2),
            "memory_value": round(memory_prediction, 2),
            "summary": summary,
            "model_info": {
                "pipeline": "modular-4stage-pipeline",
                "version": "v1.5-modular",
                "weekend_intelligence": "Prophet's built-in weekly seasonality"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting MLflow Model Registry API v2.0")
    print("🔧 Using models from MLflow Registry with MinIO fallback")
    
    # Load models on startup
    success = load_models_from_mlflow()
    if success:
        print(f"✅ Models ready from {model_source} - starting API server")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("❌ Failed to load models from MLflow/MinIO - exiting")
        exit(1)
