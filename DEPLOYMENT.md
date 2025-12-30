# Production Deployment Guide

This document outlines the strategy, requirements, and best practices for deploying the X Sentiment Analysis Pipeline to production environments.

## Table of Contents

- [Overview](#overview)
- [Infrastructure Requirements](#infrastructure-requirements)
- [Deployment Options](#deployment-options)
- [Environment Configuration](#environment-configuration)
- [Deployment Process](#deployment-process)
- [Scaling Strategy](#scaling-strategy)
- [Monitoring & Alerting](#monitoring--alerting)
- [Security Considerations](#security-considerations)
- [Cost Estimation](#cost-estimation)
- [Disaster Recovery](#disaster-recovery)
- [Troubleshooting](#troubleshooting)

## Overview

The X Sentiment Analysis Pipeline is designed as a cloud-native, containerized application suitable for deployment on major cloud platforms (AWS, GCP, Azure) or on-premises infrastructure.

### Production Readiness Checklist

- [x] Containerized with Docker
- [x] Environment-based configuration
- [x] Health check endpoints
- [x] Structured logging
- [x] Metrics instrumentation (Prometheus)
- [x] Database migrations (Alembic)
- [x] Graceful shutdown handling
- [x] Connection pooling
- [x] Error handling and retry logic
- [ ] CI/CD pipeline (optional)
- [ ] Infrastructure as Code (optional)

## Infrastructure Requirements

### Minimum Production Setup

| Component | Specification | Justification |
|-----------|--------------|---------------|
| API Servers | 2-4 vCPUs, 4-8 GB RAM | Handle concurrent requests, model inference |
| Streaming Service | 1-2 vCPUs, 2-4 GB RAM | Maintain persistent X API connection |
| Celery Workers | 2-4 vCPUs, 4-8 GB RAM (each) | Background task processing |
| PostgreSQL | 2-4 vCPUs, 8-16 GB RAM | Primary data store |
| Redis | 1-2 vCPUs, 2-4 GB RAM | Cache and message broker |
| MLflow Server | 1-2 vCPUs, 2-4 GB RAM | Experiment tracking |
| Training Compute | 8+ vCPUs, 16+ GB RAM (or GPU) | Model training tasks |

### Recommended Production Setup

| Component | Specification | Quantity | Notes |
|-----------|--------------|----------|-------|
| API Servers | 4 vCPUs, 8 GB RAM | 3+ | Behind load balancer |
| Streaming Service | 2 vCPUs, 4 GB RAM | 2+ | High availability |
| Celery Workers | 4 vCPUs, 8 GB RAM | 4+ | Separate pools by task type |
| PostgreSQL | 8 vCPUs, 32 GB RAM | 1 primary + 2 replicas | Read replicas for scaling |
| Redis | 4 vCPUs, 8 GB RAM | 1 primary + 1 replica | Sentinel for HA |
| MLflow Server | 2 vCPUs, 4 GB RAM | 1 | With S3/GCS artifact storage |
| Training Compute | GPU instance (V100/A100) | On-demand | Only during training |

### Storage Requirements

| Data Type | Initial Size | Growth Rate | Retention | Storage Type |
|-----------|-------------|-------------|-----------|--------------|
| Raw Posts | 100 GB | 10-50 GB/month | 6 months | PostgreSQL |
| Predictions | 50 GB | 5-20 GB/month | 12 months | PostgreSQL |
| Model Artifacts | 10 GB | 2-5 GB/month | Indefinite | S3/GCS/Azure Blob |
| Logs | 5 GB | 2-10 GB/month | 30 days | CloudWatch/Stackdriver |
| Metrics | 1 GB | 500 MB/month | 90 days | Prometheus/Grafana |

### Network Requirements

- **Public Internet Access**: Required for X API connectivity
- **Private Network**: VPC/VNet for internal service communication
- **Load Balancer**: For API layer (Application Load Balancer/Cloud Load Balancer)
- **NAT Gateway**: For outbound connections from private subnets
- **DNS**: For service discovery and external access
- **SSL/TLS Certificates**: Let's Encrypt or cloud provider certificates

## Deployment Options

### Option 1: Kubernetes (Recommended for Scale)

**Best For**: Large-scale deployments, multi-region, high availability requirements

**Platforms**:
- Amazon EKS (Elastic Kubernetes Service)
- Google GKE (Google Kubernetes Engine)
- Azure AKS (Azure Kubernetes Service)
- Self-managed Kubernetes

**Deployment Strategy**:
```bash
# 1. Build and push Docker images
docker build -t your-registry/sentiment-api:v1.0.0 .
docker push your-registry/sentiment-api:v1.0.0

# 2. Deploy with Helm or kubectl
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# 3. Verify deployment
kubectl get pods -n sentiment-analysis
kubectl logs -f deployment/sentiment-api -n sentiment-analysis
```

**Pros**:
- Horizontal auto-scaling
- Self-healing and rolling updates
- Service discovery and load balancing
- Multi-cloud portability
- Declarative configuration

**Cons**:
- Steeper learning curve
- Additional operational complexity
- Higher minimum resource requirements

**Estimated Cost**: $800-2000/month (depending on scale)

### Option 2: Container Platforms (AWS ECS/Fargate, Cloud Run)

**Best For**: Medium-scale deployments, simplified operations

**AWS ECS/Fargate**:
```bash
# 1. Create ECS cluster
aws ecs create-cluster --cluster-name sentiment-analysis

# 2. Create task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# 3. Create service
aws ecs create-service \
  --cluster sentiment-analysis \
  --service-name sentiment-api \
  --task-definition sentiment-api:1 \
  --desired-count 3 \
  --launch-type FARGATE
```

**Google Cloud Run**:
```bash
# Deploy directly from container
gcloud run deploy sentiment-api \
  --image gcr.io/your-project/sentiment-api:v1.0.0 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances 1 \
  --max-instances 10
```

**Pros**:
- Serverless/managed infrastructure
- Pay-per-use pricing
- Automatic scaling
- Simpler than Kubernetes

**Cons**:
- Platform lock-in
- Less control over infrastructure
- Cold start latency (Cloud Run)

**Estimated Cost**: $400-1200/month

### Option 3: Virtual Machines (EC2, Compute Engine, Azure VMs)

**Best For**: Full control, specific compliance requirements, cost optimization

**Deployment with Docker Compose**:
```bash
# 1. Provision VMs (example: AWS EC2)
# 2. Install Docker and Docker Compose
# 3. Clone repository and configure environment
# 4. Deploy services

git clone https://github.com/your-org/sentiment-analysis.git
cd sentiment-analysis
cp .env.example .env
# Edit .env with production values

docker-compose -f docker-compose.prod.yml up -d
```

**Pros**:
- Full control over infrastructure
- No container orchestration learning curve
- Can optimize costs with reserved instances
- Simple troubleshooting

**Cons**:
- Manual scaling and management
- No built-in service discovery
- Requires custom HA setup

**Estimated Cost**: $500-1500/month

### Option 4: Platform-as-a-Service (Railway, Render, Heroku)

**Best For**: Demos, small-scale deployments, rapid prototyping

**Railway Deployment**:
```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login and link project
railway login
railway link

# 3. Deploy
railway up
```

**Pros**:
- Extremely simple deployment
- Built-in PostgreSQL/Redis
- Automatic HTTPS
- Free tier or low cost

**Cons**:
- Limited scalability
- Less control over infrastructure
- Not suitable for large-scale production

**Estimated Cost**: $5-100/month

## Environment Configuration

### Required Environment Variables

Create a production `.env` file with the following variables:

```bash
# X API v2 Configuration
TWITTER_BEARER_TOKEN=your_production_bearer_token

# Database
DATABASE_URL=postgresql://user:password@prod-db.internal:5432/sentiment_prod

# Redis
REDIS_URL=redis://prod-redis.internal:6379/0

# MLflow
MLFLOW_TRACKING_URI=https://mlflow.yourcompany.com
MLFLOW_EXPERIMENT_NAME=sentiment-production

# Weights & Biases (Optional)
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=sentiment-production

# Model Configuration
MODEL_NAME=distilbert-base-uncased
MAX_LENGTH=128
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=3

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Retraining Configuration
MIN_SAMPLES_FOR_RETRAIN=5000
RETRAIN_INTERVAL_HOURS=168  # Weekly

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security
API_KEY_HEADER=X-API-Key
CORS_ORIGINS=https://yourapp.com,https://dashboard.yourapp.com
```

### Secrets Management

**AWS Secrets Manager**:
```bash
# Store secrets
aws secretsmanager create-secret \
  --name sentiment-analysis/prod \
  --secret-string file://secrets.json

# Retrieve in application
# Use boto3 or AWS SDK
```

**HashiCorp Vault**:
```bash
# Store secrets
vault kv put secret/sentiment-analysis/prod \
  twitter_bearer_token="..." \
  database_url="..."

# Retrieve in application
# Use hvac Python client
```

**Kubernetes Secrets**:
```bash
# Create secret
kubectl create secret generic sentiment-secrets \
  --from-env-file=.env.prod \
  -n sentiment-analysis

# Reference in deployment
# See k8s/deployment.yaml
```

## Deployment Process

### Pre-Deployment Checklist

- [ ] All environment variables configured
- [ ] Database migrations tested
- [ ] Health checks verified
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Backup strategy in place
- [ ] Rollback plan documented
- [ ] Monitoring and alerts configured
- [ ] Documentation updated

### Step-by-Step Deployment

#### 1. Database Setup

```bash
# Create production database
createdb -h prod-db.internal -U admin sentiment_prod

# Run migrations
export DATABASE_URL="postgresql://user:password@prod-db.internal:5432/sentiment_prod"
alembic upgrade head

# Verify schema
psql -h prod-db.internal -U admin -d sentiment_prod -c "\dt"
```

#### 2. Artifact Storage Setup

**AWS S3**:
```bash
# Create S3 bucket for MLflow artifacts
aws s3 mb s3://sentiment-mlflow-artifacts
aws s3api put-bucket-versioning \
  --bucket sentiment-mlflow-artifacts \
  --versioning-configuration Status=Enabled
```

**GCS**:
```bash
# Create GCS bucket
gsutil mb gs://sentiment-mlflow-artifacts
gsutil versioning set on gs://sentiment-mlflow-artifacts
```

#### 3. Deploy MLflow Server

```bash
# With S3 backend
mlflow server \
  --backend-store-uri postgresql://user:pass@prod-db:5432/mlflow \
  --default-artifact-root s3://sentiment-mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000

# Or deploy as container
docker run -d \
  --name mlflow \
  -p 5000:5000 \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server \
    --backend-store-uri postgresql://user:pass@prod-db:5432/mlflow \
    --default-artifact-root s3://sentiment-mlflow-artifacts \
    --host 0.0.0.0
```

#### 4. Deploy Application Services

**Using Docker Compose**:
```bash
# Pull latest images
docker-compose -f docker-compose.prod.yml pull

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Verify services
docker-compose -f docker-compose.prod.yml ps
docker-compose logs -f api
```

**Using Kubernetes**:
```bash
# Apply configurations
kubectl apply -f k8s/

# Watch rollout
kubectl rollout status deployment/sentiment-api -n sentiment-analysis

# Verify pods
kubectl get pods -n sentiment-analysis
```

#### 5. Verify Deployment

```bash
# Health check
curl https://api.yourcompany.com/health

# Test prediction
curl -X POST "https://api.yourcompany.com/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"text": "This is a test prediction"}'

# Check metrics
curl https://api.yourcompany.com/metrics
```

#### 6. Configure Load Balancer

**AWS Application Load Balancer**:
- Target Group: API instances on port 8000
- Health Check: `/health` endpoint
- HTTPS listener with ACM certificate
- Connection draining: 60 seconds

**Google Cloud Load Balancer**:
- Backend service: Instance group or Cloud Run
- Health check: HTTP on `/health`
- SSL certificate: Managed or uploaded
- Session affinity: None (stateless)

### Rolling Updates

**Zero-Downtime Deployment Strategy**:

1. **Deploy new version alongside old**:
   ```bash
   # Kubernetes rolling update
   kubectl set image deployment/sentiment-api \
     api=your-registry/sentiment-api:v1.1.0 \
     -n sentiment-analysis
   
   # Configure rolling update strategy
   kubectl patch deployment sentiment-api \
     -p '{"spec":{"strategy":{"rollingUpdate":{"maxSurge":1,"maxUnavailable":0}}}}'
   ```

2. **Health checks validate new version**
3. **Gradually shift traffic**
4. **Monitor metrics and errors**
5. **Complete rollout or rollback**

### Rollback Procedure

```bash
# Kubernetes rollback
kubectl rollout undo deployment/sentiment-api -n sentiment-analysis

# Docker Compose rollback
docker-compose -f docker-compose.prod.yml down
git checkout v1.0.0
docker-compose -f docker-compose.prod.yml up -d

# Verify rollback
curl https://api.yourcompany.com/health
```

## Scaling Strategy

### Horizontal Scaling

**API Layer**:
```bash
# Kubernetes Horizontal Pod Autoscaler
kubectl autoscale deployment sentiment-api \
  --cpu-percent=70 \
  --min=3 \
  --max=10 \
  -n sentiment-analysis

# AWS ECS Service Auto Scaling
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/sentiment-analysis/sentiment-api \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 3 \
  --max-capacity 10
```

**Celery Workers**:
```bash
# Scale based on queue depth
# Use custom metrics from Redis/Celery
kubectl scale deployment celery-worker --replicas=6 -n sentiment-analysis
```

### Vertical Scaling

**Database**:
- Read replicas for read-heavy workloads
- Increase instance size for write-heavy workloads
- Consider partitioning for very large datasets

**Redis**:
- Increase memory for larger cache
- Consider Redis Cluster for horizontal scaling

### Auto-Scaling Policies

**CPU-Based**:
```yaml
# Scale when average CPU > 70%
targetCPUUtilizationPercentage: 70
```

**Custom Metrics**:
```yaml
# Scale based on request queue length
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sentiment-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sentiment-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
```

## Monitoring & Alerting

### Metrics to Monitor

**Application Metrics** (Prometheus):
- `http_request_duration_seconds` - API latency
- `http_requests_total` - Request count by status code
- `sentiment_prediction_confidence` - Model confidence distribution
- `celery_task_queue_length` - Background task queue depth

**Infrastructure Metrics**:
- CPU utilization (target: < 70%)
- Memory utilization (target: < 80%)
- Disk utilization (target: < 85%)
- Network throughput

**Database Metrics**:
- Connection pool usage
- Query latency
- Transaction rate
- Replication lag (for replicas)

### Alert Rules

**Critical Alerts** (Immediate Response):
```yaml
# API Error Rate > 5%
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
  for: 5m
  annotations:
    summary: "High API error rate"
    
# Database Connection Pool Exhausted
- alert: DatabasePoolExhausted
  expr: postgres_connection_pool_usage > 0.95
  for: 2m
  annotations:
    summary: "Database connection pool near limit"
```

**Warning Alerts** (Monitor Closely):
```yaml
# API Latency > 1s (p95)
- alert: HighLatency
  expr: histogram_quantile(0.95, http_request_duration_seconds) > 1
  for: 10m
  
# Celery Queue Depth > 1000
- alert: CeleryQueueBacklog
  expr: celery_task_queue_length > 1000
  for: 15m
```

### Logging Strategy

**Log Aggregation**:
- AWS: CloudWatch Logs
- GCP: Cloud Logging (Stackdriver)
- Azure: Azure Monitor
- Self-hosted: ELK Stack (Elasticsearch, Logstash, Kibana)

**Structured Logging Format**:
```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "level": "INFO",
  "service": "sentiment-api",
  "trace_id": "abc123",
  "message": "Prediction completed",
  "duration_ms": 150,
  "model_version": "v1.2.0",
  "confidence": 0.92
}
```

## Security Considerations

### Network Security

1. **VPC/VNet Isolation**
   - Place databases and internal services in private subnets
   - Use security groups/firewalls to restrict access
   - API layer in public subnet behind load balancer

2. **TLS/SSL Everywhere**
   - HTTPS for all external endpoints
   - TLS for database connections
   - Certificate management with Let's Encrypt or cloud provider

3. **API Authentication**
   - API keys for service-to-service
   - JWT tokens for user authentication
   - Rate limiting per client

### Application Security

1. **Input Validation**
   - Pydantic models for request validation
   - Sanitize user inputs
   - Limit request size

2. **Secrets Management**
   - Never commit secrets to Git
   - Use secrets managers (AWS Secrets Manager, Vault)
   - Rotate credentials regularly

3. **Dependency Security**
   - Regular dependency updates
   - Security scanning (Snyk, Dependabot)
   - Container image scanning

### Data Security

1. **Encryption**
   - At rest: Database encryption, encrypted EBS volumes
   - In transit: TLS for all connections
   - Backup encryption

2. **Access Control**
   - Principle of least privilege
   - Role-based access control (RBAC)
   - Audit logging for data access

3. **Compliance**
   - GDPR: Data deletion capabilities
   - Data retention policies
   - Privacy by design

## Cost Estimation

### Small Scale (< 10K predictions/day)

| Component | Service | Monthly Cost |
|-----------|---------|--------------|
| Compute | 2 × t3.medium (AWS) | $60 |
| Database | db.t3.medium RDS | $50 |
| Redis | cache.t3.micro ElastiCache | $15 |
| Storage | 100 GB S3 | $3 |
| Load Balancer | ALB | $20 |
| X API | Basic tier | $200 |
| **Total** | | **~$350/month** |

### Medium Scale (< 100K predictions/day)

| Component | Service | Monthly Cost |
|-----------|---------|--------------|
| Compute | 4 × t3.large (API + Workers) | $240 |
| Database | db.r5.xlarge RDS + 1 replica | $400 |
| Redis | cache.r5.large ElastiCache | $150 |
| Storage | 500 GB S3 + backups | $20 |
| Load Balancer | ALB | $20 |
| X API | Pro tier | $5000 |
| Monitoring | CloudWatch/Prometheus | $50 |
| **Total** | | **~$5,880/month** |

### Large Scale (1M+ predictions/day)

| Component | Service | Monthly Cost |
|-----------|---------|--------------|
| Compute | 10 × c5.2xlarge (EKS) | $1,500 |
| Database | db.r5.4xlarge RDS + 2 replicas | $2,000 |
| Redis | cache.r5.2xlarge Cluster | $600 |
| Storage | 2 TB S3 + backups | $60 |
| Load Balancer | ALB + CloudFront CDN | $200 |
| X API | Enterprise tier | Custom |
| Monitoring | Full observability stack | $300 |
| **Total** | | **~$4,660/month + X API** |

*Note: Costs vary by region and are estimates. Always check current pricing.*

## Disaster Recovery

### Backup Strategy

**Database Backups**:
```bash
# Automated daily backups with 30-day retention
# AWS RDS: Automatic backups enabled
# Manual snapshot before major changes

# Restore from backup
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier sentiment-prod-restored \
  --db-snapshot-identifier sentiment-prod-snapshot-2025-01-15
```

**Artifact Backups**:
- S3/GCS versioning enabled
- Cross-region replication for critical artifacts
- Lifecycle policies for old versions

**Configuration Backups**:
- Infrastructure as Code in Git
- Environment configurations in secrets manager
- Regular exports of configurations

### Recovery Procedures

**Database Corruption**:
1. Stop application to prevent further corruption
2. Identify last known good backup
3. Restore database from backup
4. Verify data integrity
5. Resume application

**Service Outage**:
1. Check health endpoints and logs
2. Verify infrastructure status
3. Restart failed services
4. If persistent, rollback to previous version
5. Investigate root cause

**Data Loss**:
1. Identify scope and timing of data loss
2. Restore from most recent backup
3. Replay transactions from logs if available
4. Notify affected users
5. Implement preventive measures

### High Availability Setup

**Multi-AZ/Multi-Region**:
```yaml
# Active-Active deployment across regions
Primary Region (us-east-1):
  - Full stack deployment
  - Read-write database
  - Active-active load balancing

Secondary Region (us-west-2):
  - Full stack deployment
  - Read replica database (or read-write for active-active)
  - Active-active load balancing

Global Load Balancer:
  - Route 53 / Cloud DNS
  - Latency-based routing
  - Health checks on both regions
```

## Troubleshooting

### Common Production Issues

**High Latency**:
- Check database query performance
- Verify cache hit rate
- Monitor model inference time
- Check for network issues

**Memory Leaks**:
- Monitor container memory usage over time
- Check for unclosed connections
- Review background task cleanup
- Consider periodic container restarts

**Database Connection Exhaustion**:
- Review connection pool settings
- Check for connection leaks
- Scale database or add read replicas
- Implement connection pooling

**Failed Predictions**:
- Verify model files are accessible
- Check model version compatibility
- Review input validation errors
- Monitor model inference errors

### Debug Commands

```bash
# Check service health
curl https://api.yourcompany.com/health

# View logs
kubectl logs -f deployment/sentiment-api -n sentiment-analysis
docker-compose logs -f api

# Check database connections
psql -h prod-db -U admin -d sentiment_prod -c "SELECT count(*) FROM pg_stat_activity;"

# Check Redis
redis-cli -h prod-redis ping
redis-cli -h prod-redis info memory

# Check Celery queues
celery -A src.tasks inspect active
celery -A src.tasks inspect stats

# Test prediction
curl -X POST "https://api.yourcompany.com/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-key" \
  -d '{"text": "Test prediction"}'
```

## Continuous Improvement

### Post-Deployment Review

- Analyze deployment metrics
- Review incident reports
- Update runbooks and documentation
- Implement lessons learned
- Plan infrastructure optimizations

### Performance Optimization

- Profile slow endpoints
- Optimize database queries
- Review and tune caching strategy
- Consider CDN for static assets
- Implement request batching

### Cost Optimization

- Right-size instances based on usage
- Use reserved instances or savings plans
- Implement auto-scaling policies
- Archive old data
- Review and eliminate unused resources

## References

- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [Google Cloud Architecture Framework](https://cloud.google.com/architecture/framework)
- [The Twelve-Factor App](https://12factor.net/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [MLOps Principles](https://ml-ops.org/)
