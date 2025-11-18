# ML Predictive Scaling for Kubernetes```
modular-4stage/
├── pipelines/                        # 🔧 PIPELINE ORCHESTRATION
│   ├── modular_forecast_fixed.py     # Pipeline definition with corrected Prophet parameters
│   └── modular_forecast_test.yaml    # Generated Kubeflow pipeline YAML
├── deployment/                       # 🚀 DEPLOYMENT ARTIFACTS
│   ├── serve_kubeflow_modular.py     # Production FastAPI service
│   ├── Dockerfile.serve              # Docker build configuration
│   ├── deployment-modular.yaml       # Kubernetes deployment manifest
│   ├── requirements.txt              # Python dependencies for training
│   └── requirements-serve.txt        # Python dependencies for serving
├── components/                       # ⚙️ KUBEFLOW COMPONENTS
│   ├── data_validation_component_modular.py
│   ├── feature_engineering_component_modular.py
│   ├── model_training_component_modular.py
│   └── model_validation_component_modular.py
├── scripts/                          # 📚 REFERENCE IMPLEMENTATIONS
│   ├── data_validator.py             # DataValidator class
│   ├── feature_engineer.py           # ProphetDataPreparer class  
│   ├── model_trainer.py              # ProphetTrainer class
│   └── model_validator.py            # ModelValidator class
└── infrastructure/                   # 🏗️ INFRASTRUCTURE CONFIGS
    └── (additional infrastructure files)
```y **modular 4-stage Prophet-based ML pipeline** for predictive resource scaling in Kubernetes environments.

## 🎯 Overview

This solution provides intelligent resource forecasting using Facebook Prophet to predict CPU and memory usage patterns, enabling proactive Kubernetes resource allocation with proper ML/MLOps role separation.

## 🏗️ Modular Architecture

This pipeline implements proper separation of concerns between ML Engineers and MLOps Engineers:

- **ML Engineers**: Focus on algorithm logic in `components/` 
- **MLOps Engineers**: Focus on orchestration in `pipeline/`
- **Production API**: Ready-to-deploy FastAPI service with intelligent recommendations

## � **ACTUAL DEPLOYMENT PROCESS**

### Pipeline Training
```bash
python pipelines/modular_forecast_fixed.py
# Generates: pipelines/modular_forecast_test.yaml
```

### API Service Deployment
```bash
# Build Docker image using deployment files
cd deployment/
docker build -f Dockerfile.serve -t shivapondicherry/forecast-serve:latest .

# Deploy to Kubernetes
kubectl apply -f deployment-modular.yaml
```

**Note**: The `scripts/` directory contains standalone components for reference but are NOT used in actual deployment.

## 📁 Directory Structure

```
modular-4stage/
├── serve_kubeflow_modular.py         # 🎯 PRODUCTION API SERVICE
├── pipeline/                         # 🔧 MLOPS ENGINEER DOMAIN
│   └── modular_forecast_fixed.py     # Corrected Prophet parameters
├── components/                       # ⚙️ KUBEFLOW COMPONENTS
│   ├── data_validation_component_modular.py
│   ├── feature_engineering_component_modular.py
│   ├── model_training_component_modular.py
│   └── model_validation_component_modular.py
├── scripts/                          # � REFERENCE IMPLEMENTATIONS
│   ├── data_validator.py             # DataValidator class
│   ├── feature_engineer.py           # ProphetDataPreparer class  
│   ├── model_trainer.py              # ProphetTrainer class
│   └── model_validator.py            # ModelValidator class
├── deployment-modular.yaml           # 🚀 KUBERNETES DEPLOYMENT
├── Dockerfile.serve                  # 🐳 DOCKER BUILD CONFIG
└── modular_forecast_test.yaml        # 📋 GENERATED PIPELINE YAML
```

## ⭐ Key Features

- **Multi-timeframe forecasting**: Daily, weekly, monthly predictions
- **Conservative monthly planning**: 70% request / 150% limit multipliers
- **Standard short-term planning**: 80% request / 120% limit multipliers
- **Prophet model optimization**: Yearly seasonality + tuned changepoint detection
- **Multi-point sampling**: Improved accuracy for long-term forecasts
- **Production-quality insights**: Confidence levels and utilization analysis

## 🚀 Quick Start

### 1. Train Models (Kubeflow Pipeline)
```bash
python pipelines/modular_forecast_fixed.py
kubectl apply -f pipelines/modular_forecast_test.yaml
```

### 2. Deploy API Service
```bash
kubectl apply -f deployment/deployment-modular.yaml
kubectl port-forward -n forecast-api-modular svc/forecast-api-modular 8003:80
```

### 3. Get Predictions
```bash
# Weekly forecast
curl "http://localhost:8003/next_week"

# Monthly forecast with multi-point sampling
curl "http://localhost:8003/next_month"
```

## � Configuration

### Prophet Model Parameters (Corrected)
- `yearly_seasonality=True` - Captures annual patterns  
- `changepoint_prior_scale=0.05` - Balanced trend detection
- `daily_seasonality=True` - Handles daily cycles

### Recommendation Logic
- **Weekly**: 80% request, 120% limit (predictable short-term)
- **Monthly**: 70% request, 150% limit (conservative long-term planning)

## 📈 API Endpoints

| Endpoint | Description | Multipliers | Use Case |
|----------|-------------|-------------|----------|
| `/next_day` | Tomorrow's forecast | 80% / 120% | Daily planning |
| `/next_week` | Next week average | 80% / 120% | Sprint planning |
| `/next_month` | Monthly with 4-week sampling | 70% / 150% | Capacity planning |
| `/health` | Service health check | - | Monitoring |

## 🧪 Validation Results

- **Prediction Accuracy**: ~0.16 CPU cores (corrected from 0.015)
- **Model Performance**: Production-validated Prophet parameters
- **Deployment**: Successfully running with proper multipliers
- **Multi-point Sampling**: Improved monthly forecast reliability

## 🛠️ Development

### Local Testing with Virtual Environment
```bash
source ~/venv/kfp/bin/activate
cd modular-4stage
python scripts/serve_kubeflow_modular.py
```

### Docker Build
```bash
docker build -f Dockerfile.serve -t shivapondicherry/forecast-serve:v1.5-modular-final .
```

## 🚀 4-Stage Pipeline

### Stage 1: Data Validation
- **Component**: `data_validation_component_modular.py`
- **Script**: `scripts/data_validator.py` → `DataValidator` class
- **Purpose**: Validates CSV data from MinIO storage

### Stage 2: Feature Engineering  
- **Component**: `feature_engineering_component_modular.py`
- **Script**: `scripts/feature_engineer.py` → `ProphetDataPreparer` class
- **Purpose**: Prepares Prophet-optimized features (timestamp + metrics)

### Stage 3: Model Training
- **Component**: `model_training_component_modular.py` 
- **Script**: `scripts/model_trainer.py` → `ProphetTrainer` class
- **Purpose**: Trains Prophet models with real MinIO data

### Stage 4: Model Validation
- **Component**: `model_validation_component_modular.py`
- **Script**: `scripts/model_validator.py` → `ModelValidator` class  
- **Purpose**: Validates trained models before deployment

## 🔒 Production Safety

- **Separate Model Names**: Uses `modular-` prefix
  - Production: `models/cpu_prophet_model.pkl`
  - Modular: `models/modular-cpu_prophet_model.pkl`
- **No Disruption**: Won't affect existing production models
- **Independent Stages**: Each stage runs independently

## 📊 Prophet Model Intelligence

The pipeline leverages Prophet's built-in capabilities:
- **Automatic Seasonality**: Prophet detects weekly/daily patterns
- **No Manual Features**: Time features unnecessary (Prophet uses timestamps)
- **Weekend Intelligence**: `/next_weekend` endpoint works via Prophet's timestamp analysis

## 🏃‍♂️ Running the Pipeline

### Option 1: Kubeflow UI
1. Upload `modular_pipeline.yaml`
2. Monitor each stage separately
3. Check model paths: `models/modular-*`

### Option 2: kubectl
```bash
kubectl apply -f modular_pipeline.yaml -n kubeflow
kubectl get workflows -n kubeflow
```

## 🧪 Testing Strategy

1. **Local Testing**: Each script can be tested independently
2. **Component Testing**: Components have fallback logic
3. **Pipeline Testing**: Full 4-stage execution with separate model names

## 📝 Lessons Learned Applied

- ✅ Pre-built Docker images (`shivapondicherry/forecast-train:latest`)
- ✅ Independent stage execution (no parameter passing)
- ✅ Real MinIO data (not synthetic)
- ✅ ASCII naming only (no special characters)
- ✅ Proper error handling with fallbacks
- ✅ Separate model naming for safety

## 🎯 Benefits of Modular Approach

1. **Clear Responsibilities**: ML vs MLOps domain separation
2. **Reusable Components**: Scripts can be used in different pipelines
3. **Independent Development**: ML Engineers work on algorithms, MLOps on orchestration
4. **Easy Testing**: Each layer can be tested separately
5. **Production Safety**: Parallel development without disruption

## 🔍 Key Files

- `scripts/*.py`: Pure ML logic (ML Engineer domain)
- `components/*_modular.py`: Kubeflow wrappers (MLOps Engineer domain)
- `pipeline/modular_pipeline.py`: Pipeline orchestration (MLOps Engineer domain)
- `modular_pipeline.yaml`: Deployment configuration (DevOps domain)

## 🏢 Production Considerations

- **Resource Planning**: Different multipliers for different time horizons
- **Uncertainty Handling**: Higher limits for longer-term forecasts
- **Multi-point Sampling**: Reduces single-point prediction errors for monthly forecasts
- **Kubernetes Integration**: Ready-to-use resource specifications
- **Model Safety**: Separate `modular-` prefixed models (won't affect existing production)

## 📊 Model Performance

The corrected Prophet model delivers consistent predictions:
- **CPU Forecasting**: ~0.16 cores (vs previous 0.015 error)
- **Memory Prediction**: GB-scale accuracy with trend detection  
- **Confidence Intervals**: Uncertainty quantification included
- **Seasonal Awareness**: Captures weekly/daily patterns automatically

## 🔄 CI/CD Integration

Ready for integration with GitOps workflows:
- Kubeflow pipeline automation
- Docker image versioning (`v1.5-modular-final`)
- Kubernetes deployment manifests
- Health check endpoints for monitoring

---

---

**Built for Production** | **Modular by Design** | **ML/MLOps Separation** | **Prophet-Powered**
