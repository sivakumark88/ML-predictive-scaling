# ML Predictive Scaling Architecture

## System Overview

This document describes the architecture of the ML Predictive Scaling system for Kubernetes resource forecasting using Prophet time series models.

## 🚀 Data Flow Overview (Start Here!)

**For newcomers**: Follow this numbered sequence to understand how the system works:

1. **① Data Collection**: Kubernetes cluster generates resource metrics → Prometheus collects them
2. **② Data Storage**: Historical metrics stored in MinIO as CSV files  
3. **③ ML Training**: 4-stage pipeline processes data and trains Prophet models
4. **④ Model Registry**: Trained models are registered and versioned in MLflow Model Registry
5. **⑤ API Serving**: FastAPI service loads models directly from MLflow Model Registry and provides predictions
6. **⑥ Consumption**: DevOps teams, Platform Engineers, and Automation tools consume forecasts

## Architecture Diagram

```mermaid
graph LR
    %% Data Sources (Left)
    subgraph "📊 Data Sources"
        K8S["<img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kubernetes/kubernetes-plain.svg' width='40' height='40'/><br/>⚙️ Kubernetes Cluster<br/>Resource Metrics"]
        PROM["<img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/prometheus/prometheus-original.svg' width='40' height='40'/><br/>📈 Prometheus<br/>Monitoring & Metrics"]
        K8S --> PROM
    end

    %% Data Storage (Center-Left)
  subgraph "💾 Data Storage"
<<<<<<< HEAD
    MINIO[🗄️ MinIO Object Storage<br/>📁 mlpipeline bucket]
=======
    MINIO["<img src='https://blog.min.io/content/images/size/w2000/2019/05/0_hReq8dEVSFIYJMDv.png' width='40' height='40'/><br/>🗄️ MinIO Object Storage<br/>📁 mlpipeline bucket"]
>>>>>>> 23c72a13c18db8d8c174160808faaad50861f2dc
    DATASET[📄 metrics_dataset.csv]
  end

  subgraph "📦 Model Registry"
<<<<<<< HEAD
    MLFLOW[📦 MLflow Model Registry<br/>Versioned Prophet Models]
=======
    MLFLOW["<img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRy8RXezmKJEzVNsbt52H8__bwBgXk6mjC7CA&s' width='40' height='40'/><br/>📦 MLflow Model Registry<br/>Versioned Prophet Models"]
>>>>>>> 23c72a13c18db8d8c174160808faaad50861f2dc
    MODELS[🧠 Prophet Models<br/>modular-cpu-prophet-model<br/>modular-memory-prophet-model]
  end

    %% ML Pipeline (Center)
    subgraph "🔬 ML Pipeline - Kubeflow"
        DV["<img src='https://avatars.githubusercontent.com/u/33164907?s=200&v=4' width='50' height='50' style='object-fit: contain;'/><br/>1️⃣ Data Validation<br/>📋 CSV validation<br/>🔍 Quality checks"]
        FE["<img src='https://avatars.githubusercontent.com/u/33164907?s=200&v=4' width='50' height='50' style='object-fit: contain;'/><br/>2️⃣ Feature Engineering<br/>🛠️ Prophet prep<br/>📈 Time series format"]
        MT["<img src='https://avatars.githubusercontent.com/u/33164907?s=200&v=4' width='50' height='50' style='object-fit: contain;'/><br/>3️⃣ Model Training<br/>🧠 CPU/Memory models<br/>⚙️ Prophet params"]
        MV["<img src='https://avatars.githubusercontent.com/u/33164907?s=200&v=4' width='50' height='50' style='object-fit: contain;'/><br/>4️⃣ Model Validation<br/>✅ Performance checks<br/>📊 Forecast validation"]
        
        DV --> FE
        FE --> MT
        MT --> MV
    end

    %% Kubernetes Infrastructure (Center-Right)
    subgraph "☸️ Kubernetes Deployment"
        subgraph "🏗️ Training Infrastructure"
            KF_NS["<img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kubernetes/kubernetes-plain.svg' width='30' height='30'/><br/>🔬 kubeflow namespace<br/>🔄 Pipeline execution"]
            TRAIN_IMG[🐳 Docker Image:<br/>shivapondicherry/forecast-train:v6-registry-fix<br/>📦 Contains: Python, Prophet, Kubeflow SDK]
        end
        
        subgraph "🚀 Serving Infrastructure"  
            API_NS["<img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kubernetes/kubernetes-plain.svg' width='30' height='30'/><br/>🌐 forecast-api-modular namespace<br/>🚀 API deployment"]
            SERVE_IMG[🐳 Docker Image:<br/>shivapondicherry/forecast-serve:mlflow<br/>📦 Contains: FastAPI, Prophet models, uvicorn]
        end
    end

    %% API Layer (Right)
    subgraph "🌐 API Endpoints"
        API["<img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/fastapi/fastapi-original.svg' width='40' height='40'/><br/>⚡ FastAPI Service<br/>🌐 Production API"]
        DAY[📅 next_day<br/>Tomorrow<br/>80/120%]
        WEEK[📅 next_week<br/>Weekly<br/>80/120%] 
        MONTH[📅 next_month<br/>Monthly<br/>70/150%]
        HEALTH[💚 health<br/>Status Check]
        
        API --> DAY
        API --> WEEK
        API --> MONTH
        API --> HEALTH
    end

    %% Consumers (Far Right)
    subgraph "👥 Consumers"
        DEVOPS[👷 DevOps Teams<br/>📊 Capacity planning]
        PLATFORM[🔧 Platform Engineers<br/>⚙️ Resource optimization] 
        AUTO[🤖 Automation Tools<br/>📈 Auto-scaling systems]
    end

    %% Main Data Flow (Horizontal) - Numbered sequence for clarity
    K8S -->|"①<br/>Live metrics"| PROM
    PROM -.->|"②<br/>Historical data"| DATASET
    DATASET -->|"③<br/>Training data"| DV
  MV -->|"④<br/>Register models"| MLFLOW
  MLFLOW -->|"⑤<br/>Load models"| API
    API -->|"⑥<br/>Predictions"| DEVOPS
    API -->|"⑥<br/>Predictions"| PLATFORM
    API -->|"⑥<br/>Predictions"| AUTO
    
    %% Storage connections
  MINIO --> DATASET
    
    %% Deployment connections (Kubernetes orchestrates pipeline stages)
    KF_NS --> DV
    KF_NS --> FE
    KF_NS --> MT
    KF_NS --> MV
    API_NS --> API
    
    %% Docker runtime connections (shows what runs where)
    TRAIN_IMG --> DV
    TRAIN_IMG --> FE
    TRAIN_IMG --> MT
    TRAIN_IMG --> MV
    SERVE_IMG --> API

    %% Consumer connections
    API --> DEVOPS
    API --> PLATFORM
    API --> AUTO

    %% Styling with better colors for each technology
    classDef dataSource fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1
    classDef storage fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#4a148c
    classDef pipeline fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    classDef serving fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#bf360c
    classDef infra fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#880e4f
    classDef client fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#33691e
    classDef docker fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:#01579b

    class K8S,PROM dataSource
    class MINIO,DATASET,MODELS storage
    class DV,FE,MT,MV pipeline
    class API,DAY,WEEK,MONTH,HEALTH serving
    class KF_NS,API_NS infra
    class DEVOPS,PLATFORM,AUTO client
    class TRAIN_IMG,SERVE_IMG docker
```

## Key Components

### 📊 Data Sources
- **⚙️ Kubernetes Cluster**: Provides real-time resource metrics (CPU, memory usage)
- **📈 Prometheus**: Collects and aggregates Kubernetes metrics with time series data

### 💾 Data Storage  
- **🗄️ MinIO**: Object storage for datasets in `mlpipeline` bucket
- **📄 CSV Dataset**: Historical metrics data for training Prophet models

### 📦 Model Registry
- **📦 MLflow Model Registry**: Central registry for versioned Prophet models
- **🧠 Prophet Models**: Registered ML models for CPU and memory forecasting

### 🔬 ML Pipeline (Kubeflow)
- **1️⃣ Data Validation**: Validates CSV data quality and format
- **2️⃣ Feature Engineering**: Prepares time series data for Prophet training
- **3️⃣ Model Training**: Trains separate Prophet models for CPU and memory forecasting
- **4️⃣ Model Validation**: Validates model performance and accuracy

### ☸️ Kubernetes Infrastructure
- **🏗️ Training Infrastructure**: 
  - Namespace: `kubeflow`
  - Container: `sivakumark88/forecast-train:latest`
  - Purpose: Runs the 4-stage ML pipeline
  
- **🚀 Serving Infrastructure**:
  - Namespace: `forecast-api-modular`
  - Container: `shivapondicherry/forecast-serve:mlflow`
  - Purpose: Hosts the FastAPI prediction service, loads models from MLflow Model Registry

### 🌐 API Endpoints
- **⚡ FastAPI Service**: Production-ready REST API for resource forecasting
- **📅 Prediction Endpoints**:
  - `next_day`: Tomorrow's forecast (80-120% confidence)
  - `next_week`: Weekly forecast (80-120% confidence) 
  - `next_month`: Monthly forecast (70-150% confidence)
  - `health`: Service health check

### 👥 Consumer Applications
- **👷 DevOps Teams**: Use forecasts for capacity planning and resource allocation
- **🔧 Platform Engineers**: Optimize cluster resources based on predictions
- **🤖 Automation Tools**: Implement auto-scaling based on forecast recommendations

## Technical Architecture Details

### Docker Images
- **Training Image**: `sivakumark88/forecast-train:latest`
  - Contains Python, Prophet, Kubeflow SDK
  - Runs ML pipeline stages in Kubeflow
  
- **Serving Image**: `shivapondicherry/forecast-serve:mlflow`
  - Contains FastAPI, MLflow client, uvicorn
  - Loads models from MLflow Model Registry and provides REST API for predictions

### Data Flow
1. **📈 Metrics Collection**: Kubernetes → Prometheus → MinIO storage
2. **🔬 ML Training**: CSV data → 4-stage pipeline → Trained models
3. **📦 Model Registration**: Trained models → MLflow Model Registry (versioned)
4. **🌐 Prediction Serving**: Models (from MLflow) → FastAPI → JSON predictions
5. **👥 Consumption**: API → DevOps/Platform teams → Resource decisions

### Prediction Logic
- **Daily/Weekly**: 80-120% multipliers (conservative for short-term)
- **Monthly**: 70-150% multipliers (wider range for long-term uncertainty)
- **Prophet Configuration**: `yearly_seasonality=True`, `changepoint_prior_scale=0.05`

## Deployment Commands

### Build and Deploy Training Pipeline
```bash
# Generate pipeline YAML
python pipelines/modular_forecast_fixed.py

# Deploy to Kubeflow
kubectl apply -f pipelines/modular_forecast_test.yaml -n kubeflow
```

### Build and Deploy API Service (MLflow)
```bash
# Build serving image
docker build -f deployment/Dockerfile.serve -t shivapondicherry/forecast-serve:mlflow .

# Push image to registry
docker push shivapondicherry/forecast-serve:mlflow

# Deploy to Kubernetes
kubectl apply -f deployment/deployment-modular.yaml
```

## Monitoring and Observability

### Health Checks
- **API Health**: `GET /health` endpoint provides service status
- **Model Validation**: Pipeline stage 4 validates prediction quality
- **Kubernetes Health**: Standard K8s pod/deployment monitoring

### Logging
- **Pipeline Logs**: Available via `kubectl logs` in kubeflow namespace
- **API Logs**: Available via `kubectl logs` in forecast-api-modular namespace
- **Prometheus Metrics**: Standard FastAPI and Kubernetes metrics

This architecture provides a robust, scalable ML system for Kubernetes resource forecasting with clear separation of concerns between training and serving components.
