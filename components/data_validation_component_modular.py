#!/usr/bin/env python3
"""
MLOps Component: Data Validation Component (PROPER MODULAR VERSION)
This component imports and uses ML Engineer's DataValidator from scripts/
"""

from kfp.dsl import component

# Using pre-built image with all packages (avoiding pip timeouts)
DOCKER_IMAGE = "shivapondicherry/forecast-train:latest"

@component(base_image=DOCKER_IMAGE)
def data_validation_component() -> dict:
    """
    Stage 1: Data Validation Component
    Uses ML Engineer's DataValidator class from scripts/data_validator.py
    """
    import sys
    import os
    import json
    
    print("=== STAGE 1: DATA VALIDATION (MODULAR APPROACH) ===")
    
    try:
        # Add paths to import ML Engineer's scripts
        sys.path.append('/opt/mlpipeline/launch')
        sys.path.append('.')
        sys.path.append('./scripts')
        
        # Import ML Engineer's DataValidator class
        from data_validator import DataValidator
        
        print("✅ Successfully imported ML Engineer's DataValidator")
        
        # Use ML Engineer's validation logic
        validator = DataValidator(min_records=1000)
        
        # Run validation (loads real MinIO data internally)
        validation_results = validator.validate_dataset()
        
        print(f"✅ ML Engineer's validation complete: {validation_results['validation_status']}")
        return validation_results
        
    except ImportError as e:
        print(f"⚠️  Could not import ML Engineer's script: {e}")
        print("📦 Using fallback validation logic...")
        
        # Fallback: Direct validation (same logic as scripts/data_validator.py)
        from minio import Minio
        import pandas as pd
        
        # MinIO connection
        minio_client = Minio(
            "minio-service.kubeflow.svc.cluster.local:9000",
            access_key="minio",
            secret_key="minio123", 
            secure=False
        )
        
        print("📥 Downloading metrics dataset from MinIO")
        minio_client.fget_object("mlpipeline", "metrics_dataset.csv", "metrics_dataset.csv")
        
        # Apply validation logic (from ML Engineer's script)
        df = pd.read_csv("metrics_dataset.csv")
        print(f"✅ Dataset loaded: {len(df)} records")
        print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        missing_data = df.isnull().sum().sum()
        required_columns = ['timestamp', 'cpu_mean_5m', 'mem_mean_5m']
        has_required_cols = all(col in df.columns for col in required_columns)
        
        validation_status = "PASSED" if (missing_data == 0 and len(df) > 1000 and has_required_cols) else "FAILED"
        
        print(f"📤 Component validation: {validation_status}")
        
        return {
            'record_count': len(df),
            'date_range': {
                'start': str(df['timestamp'].min()),
                'end': str(df['timestamp'].max())
            },
            'columns': list(df.columns),
            'missing_values': int(missing_data),
            'validation_status': validation_status
        }

if __name__ == "__main__":
    print("Data Validation Component - uses ML Engineer's scripts/data_validator.py")
