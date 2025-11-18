#!/usr/bin/env python3
"""
Simple Feature Engineering Script
Processes CSV for Prophet forecasting
"""

import pandas as pd
from minio import Minio

class FeatureEngineer:
    """Simple feature engineering for Prophet"""
    
    def __init__(self):
        self.minio_client = Minio(
            "minio-service.kubeflow.svc.cluster.local:9000",
            access_key="minio", 
            secret_key="minio123", 
            secure=False
        )
        
    def prepare_data(self) -> pd.DataFrame:
        """Load and prepare Prophet-ready data"""
        print("Loading metrics_dataset.csv for feature engineering...")
        
        # Load raw data from MinIO
        self.minio_client.fget_object("mlpipeline", "metrics_dataset.csv", "metrics_dataset.csv")
        df = pd.read_csv("metrics_dataset.csv")
        
        print(f"Raw data: {len(df)} records")
        
        # Clean and prepare for Prophet
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Select only required columns for Prophet
        prophet_data = df[['timestamp', 'cpu_mean_5m', 'mem_mean_5m']].copy()
        prophet_data = prophet_data.dropna().reset_index(drop=True)
        
        print(f"Prophet-ready data: {len(prophet_data)} records")
        
        # Save processed data to MinIO
        prophet_data.to_csv("prophet_ready_data.csv", index=False)
        self.minio_client.fput_object("mlpipeline", "prophet_ready_data.csv", "prophet_ready_data.csv")
        print("Processed data saved to MinIO")
        
        return prophet_data

def run_feature_engineering():
    """Main function to run feature engineering"""
    engineer = FeatureEngineer()
    return engineer.prepare_data()

if __name__ == "__main__":
    data = run_feature_engineering()
    print(f"Feature engineering complete: {len(data)} records")
