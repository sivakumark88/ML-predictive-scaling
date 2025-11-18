#!/usr/bin/env python3
"""
Simple Data Validation Script
Validates CSV dataset from MinIO
"""

import pandas as pd
from minio import Minio
from typing import Dict, Any

class DataValidator:
    """Simple data validation for forecasting pipeline"""
    
    def __init__(self, min_records: int = 1000):
        self.min_records = min_records
        self.minio_client = Minio(
            "minio-service.kubeflow.svc.cluster.local:9000",
            access_key="minio", 
            secret_key="minio123", 
            secure=False
        )
        
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate metrics_dataset.csv from MinIO"""
        print("Loading metrics_dataset.csv from MinIO...")
        
        # Load data from MinIO
        self.minio_client.fget_object("mlpipeline", "metrics_dataset.csv", "metrics_dataset.csv")
        df = pd.read_csv("metrics_dataset.csv")
        
        print(f"Dataset loaded: {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        
        # Validation checks
        required_columns = ['timestamp', 'cpu_mean_5m', 'mem_mean_5m']
        missing_columns = [col for col in required_columns if col not in df.columns]
        missing_values = df.isnull().sum().sum()
        
        # Determine validation status
        if len(df) >= self.min_records and missing_values == 0 and not missing_columns:
            validation_status = 'PASSED'
            print("Data validation PASSED")
        else:
            validation_status = 'FAILED'
            print("Data validation FAILED:")
            print(f"   Records: {len(df)} (need {self.min_records})")
            print(f"   Missing values: {missing_values}")
            print(f"   Missing columns: {missing_columns}")
        
        return {
            'validation_status': validation_status,
            'record_count': len(df),
            'missing_values': int(missing_values),
            'missing_columns': missing_columns,
            'columns': list(df.columns)
        }

def run_validation():
    """Main function to run validation"""
    validator = DataValidator()
    return validator.validate_dataset()

if __name__ == "__main__":
    result = run_validation()
    print(f"Validation result: {result}")
