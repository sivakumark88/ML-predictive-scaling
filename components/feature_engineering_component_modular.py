#!/usr/bin/env python3
"""
MLOps Component: Feature Engineering Component (PROPER MODULAR VERSION)
This component imports and uses ML Engineer's ProphetDataPreparer from scripts/
"""

from kfp.dsl import component

# Using pre-built image
DOCKER_IMAGE = "shivapondicherry/forecast-train:latest"

@component(base_image=DOCKER_IMAGE)
def feature_engineering_component() -> dict:
    """
    Stage 2: Feature Engineering Component
    Uses ML Engineer's ProphetDataPreparer from scripts/feature_engineer.py
    """
    import sys
    import os
    import json
    
    print("=== STAGE 2: FEATURE ENGINEERING (MODULAR APPROACH) ===")
    
    try:
        # Add paths to import ML Engineer's scripts
        sys.path.append('/opt/mlpipeline/launch')
        sys.path.append('.')
        sys.path.append('./scripts')
        
        # Import ML Engineer's ProphetDataPreparer class
        from feature_engineer import ProphetDataPreparer
        
        print("✅ Successfully imported ML Engineer's ProphetDataPreparer")
        
        # Use ML Engineer's feature engineering logic
        preparer = ProphetDataPreparer()
        
        # Prepare Prophet-ready data (loads MinIO data internally)
        prophet_data = preparer.prepare_prophet_data()
        
        # Get data summary
        summary = preparer.get_data_summary(prophet_data)
        
        print(f"✅ ML Engineer's feature engineering complete: {summary['total_records']} records")
        
        # Save Prophet-ready data for next stage
        prophet_data.to_csv("prophet_ready_data.csv", index=False)
        
        # Upload to MinIO for next stage
        from minio import Minio
        minio_client = Minio(
            "minio-service.kubeflow.svc.cluster.local:9000",
            access_key="minio",
            secret_key="minio123", 
            secure=False
        )
        
        print("📤 Uploading Prophet-ready data to MinIO")
        minio_client.fput_object("mlpipeline", "modular-prophet_ready_data.csv", "prophet_ready_data.csv")
        
        return summary
        
    except ImportError as e:
        print(f"⚠️  Could not import ML Engineer's script: {e}")
        print("📦 Using fallback feature engineering...")
        
        # Fallback: Direct Prophet preparation
        from minio import Minio
        import pandas as pd
        
        minio_client = Minio(
            "minio-service.kubeflow.svc.cluster.local:9000",
            access_key="minio",
            secret_key="minio123", 
            secure=False
        )
        
        print("📥 Downloading metrics dataset")
        minio_client.fget_object("mlpipeline", "metrics_dataset.csv", "metrics_dataset.csv")
        
        # Prophet-ready data preparation (from ML Engineer's logic)
        df = pd.read_csv("metrics_dataset.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Keep only what Prophet needs
        prophet_data = df[['timestamp', 'cpu_mean_5m', 'mem_mean_5m']].copy()
        prophet_data = prophet_data.dropna().reset_index(drop=True)
        
        print(f"✅ Prophet-ready data: {len(prophet_data)} records")
        
        # Save for next stage
        prophet_data.to_csv("prophet_ready_data.csv", index=False)
        minio_client.fput_object("mlpipeline", "modular-prophet_ready_data.csv", "prophet_ready_data.csv")
        
        return {
            'total_records': len(prophet_data),
            'columns': list(prophet_data.columns),
            'date_range': {
                'start': str(prophet_data['timestamp'].min()),
                'end': str(prophet_data['timestamp'].max())
            },
            'prophet_optimized': True
        }

if __name__ == "__main__":
    print("Feature Engineering Component - uses ML Engineer's scripts/feature_engineer.py")
