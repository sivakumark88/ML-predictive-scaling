#!/bin/bash
# Build and Deploy ML Predictive Scaling System

set -e

echo "🏗️ Building ML Predictive Scaling Docker Images"

# Build training image (contains ML pipeline components)
echo "📦 Building training image..."
docker build -f deployment/Dockerfile.train -t sivakumark88/forecast-train:latest .

# Build serving image (contains FastAPI service)  
echo "📦 Building serving image..."
docker build -f deployment/Dockerfile.serve -t sivakumark88/forecast-serve:v1.5-modular-improved .

echo "🚀 Pushing images to registry..."
docker push sivakumark88/forecast-train:latest
docker push sivakumark88/forecast-serve:v1.5-modular-improved

echo "🔬 Generating and deploying ML pipeline..."
cd pipelines
python modular_forecast_fixed.py
kubectl apply -f modular_forecast_test.yaml -n kubeflow

echo "🌐 Deploying API service..."
kubectl set image deployment/forecast-api-modular -n forecast-api-modular \
  forecast-api-modular=sivakumark88/forecast-serve:v1.5-modular-improved

echo "✅ Deployment complete!"
echo ""
echo "📊 Check pipeline status:"
echo "kubectl get workflows -n kubeflow"
echo ""
echo "🌐 Check API status:"
echo "kubectl get pods -n forecast-api-modular"
