# Kubeflow Basics - The 3 Essential Patterns

## 🎯 **MLOps Role: Wrap ML Engineer's Code with Kubeflow**

**The 3 Core Patterns You Need**:
1. **Component** = Wrap one ML script 
2. **Pipeline** = Connect components in order
3. **File I/O** = Pass data between components using `with open()`

---

## 🚀 **Essential Imports**

```python
from kfp.v2.dsl import component, pipeline      # Create components & pipelines
from kfp.v2.dsl import InputPath, OutputPath    # Handle files between steps
from kfp.v2 import compiler                     # Convert to YAML
```

---

## 🔧 **1. Component Pattern (Wrapping ML Engineer's Code)**

```python
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy"]
)
def data_validation_component(
    input_data: InputPath('Dataset'),
    output_data: OutputPath('Dataset')
):
    """Wrap ML Engineer's data validation script"""
    import pandas as pd
    import json
    
    # Import ML Engineer's code (don't modify it!)
    from scripts.data_validator import DataValidator
    
    # Execute ML logic
    validator = DataValidator()
    df = pd.read_csv(input_data)
    results = validator.validate_dataset(df)
    
    # Save results for next component
    df.to_csv(output_data, index=False)
    
    return results
```

---

## 🏗️ **2. Pipeline Pattern (Connecting Components)**

```python
@pipeline(name='predictive-scaling-pipeline')
def predictive_scaling_pipeline():
    """Connect all ML components in order"""
    
    # Stage 1: Data validation
    data_task = data_validation_component()
    
    # Stage 2: Feature engineering (uses stage 1 output)
    feature_task = feature_engineering_component(
        input_data=data_task.outputs['output_data']
    )
    
    # Stage 3: Model training (uses stage 2 output)
    training_task = model_training_component(
        input_data=feature_task.outputs['output_data']
    )
    
    # Stage 4: Model validation (uses stage 3 output)
    validation_task = model_validation_component(
        input_model=training_task.outputs['trained_model']
    )
```

---

## 📁 **3. File I/O Pattern (Data Flow Between Components)**

```python
# Reading from previous component
with open(input_data, 'r') as f:
    data = pd.read_csv(f)

# Saving for next component  
with open(output_data, 'w') as f:
    data.to_csv(f, index=False)

# JSON for configuration/results
with open(output_config, 'w') as f:
    json.dump({'accuracy': 0.95}, f)
```

---

## 🚀 **Deployment (Python → YAML)**

```python
from kfp.v2 import compiler

compiler.Compiler().compile(
    pipeline_func=predictive_scaling_pipeline,
    package_path='pipeline.yaml'
)
```

**That's it! These 3 patterns handle most MLOps work.**

---

## ✅ **Your Project Structure**

```
modular-4stage/
├── scripts/           # ML Engineer's pure Python code
├── components/        # Your MLOps wrappers  
└── pipeline/          # Your MLOps orchestration
```

**Perfect for beginners - master these basics first!**
