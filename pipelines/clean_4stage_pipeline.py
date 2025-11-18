#!/usr/bin/env python3
"""
Clean 4-Stage MLflow Pipeline
Uses simple scripts from Docker image with MLflow-only storage
"""

import kfp
from kfp import dsl
from kfp.dsl import component

DOCKER_IMAGE = "shivapondicherry/forecast-train:v6-registry-fix"  # Fixed MLflow Registry artifact downloading

@component(base_image=DOCKER_IMAGE)
def data_validation_component():
    """Stage 1: Validate CSV dataset from MinIO"""
    print("=== STAGE 1: DATA VALIDATION ===")
    
    # Scripts are available in the Docker image at /app/scripts via PYTHONPATH
    from simple_data_validator import run_validation
    
    validation_results = run_validation()
    
    if validation_results['validation_status'] != 'PASSED':
        raise Exception("Data validation failed - stopping pipeline")
    
    print(f"Stage 1 Complete: {validation_results['record_count']} records validated")

@component(base_image=DOCKER_IMAGE)
def feature_engineering_component():
    """Stage 2: Feature engineering on CSV"""
    print("=== STAGE 2: FEATURE ENGINEERING ===")
    
    from simple_feature_engineer import run_feature_engineering
    
    processed_data = run_feature_engineering()
    
    print(f"Stage 2 Complete: {len(processed_data)} records processed for Prophet")

@component(base_image=DOCKER_IMAGE)
def model_training_component():
    """Stage 3: Train models and upload to MLflow ONLY"""
    import subprocess
    import sys
    
    print("=== STAGE 3: MODEL TRAINING & MLFLOW UPLOAD ===")
    
    # Install MLflow
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mlflow==2.16.0", "--quiet"])
    
    from simple_model_trainer import run_training
    
    cpu_run_id, memory_run_id = run_training()
    
    print("Stage 3 Complete:")
    print(f"   CPU Model Run: {cpu_run_id}")
    print(f"   Memory Model Run: {memory_run_id}")
    print("   Models uploaded to MLflow Registry and promoted to Staging")

@component(base_image=DOCKER_IMAGE)
def model_validation_component():
    """Stage 4: Validate models from MLflow ONLY"""
    import subprocess
    import sys
    
    print("=== STAGE 4: MODEL VALIDATION ===")
    
    # Install MLflow
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mlflow==2.16.0", "--quiet"])
    
    from simple_model_validator import run_validation
    
    validation_results = run_validation()
    
    if validation_results['overall_status'] == 'PASSED':
        print("All models validated and promoted to Production!")
    else:
        print("Model validation failed - models not promoted")
        raise Exception("Model validation failed")
    
    print(f"Stage 4 Complete: Validation {validation_results['overall_status']}")

@dsl.pipeline(
    name="clean-4stage-mlflow-pipeline",
    description="Clean 4-stage pipeline using Docker image with scripts: validate -> engineer -> train -> validate (MLflow only)"
)
def clean_4stage_pipeline():
    """
    Clean 4-stage ML pipeline using Docker image with embedded scripts:
    1. Validate dataset CSV from MinIO
    2. Feature engineering on CSV
    3. Train models -> MLflow Registry (Staging)
    4. Validate models -> MLflow Registry (Production)
    """
    
    # Stage 1: Data Validation
    validation_task = data_validation_component().set_display_name('Stage 1: Data Validation')\
        .set_caching_options(False)
    
    # Stage 2: Feature Engineering  
    engineering_task = feature_engineering_component().set_display_name("Stage 2: Feature Engineering")\
        .set_caching_options(False)
    engineering_task.after(validation_task)
    
    # Stage 3: Model Training
    training_task = model_training_component().set_display_name("Stage 3: Model Training to MLflow")\
        .set_caching_options(False)
    training_task.after(engineering_task)
    
    # Stage 4: Model Validation
    model_validation_task = model_validation_component().set_display_name("Stage 4: Model Validation to Production")\
        .set_caching_options(False)
    model_validation_task.after(training_task)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=clean_4stage_pipeline,
        package_path="clean_4stage_pipeline.yaml"
    )
    
    print("Clean 4-Stage Pipeline compiled successfully!")
    print("File: clean_4stage_pipeline.yaml")
    print("Docker Image: shivapondicherry/forecast-train:v6-registry-fix")
    print("Flow: CSV Validation -> Feature Engineering -> MLflow Training -> MLflow Validation")
    print("Storage: MLflow Registry ONLY (no MinIO fallback)")
    print("Scripts: Embedded in Docker image at /app/scripts")
    print("Ready to upload pipeline to Kubeflow UI!")
