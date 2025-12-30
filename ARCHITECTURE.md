# System Architecture

This document provides a comprehensive overview of the X Sentiment Analysis Pipeline architecture, including component interactions, data flow, technology choices, and design decisions.

## Table of Contents

- [Overview](#overview)
- [System Architecture Diagram](#system-architecture-diagram)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Design Decisions](#design-decisions)
- [Scalability Considerations](#scalability-considerations)
- [Security Architecture](#security-architecture)

## Overview

The X Sentiment Analysis Pipeline is a production-ready MLOps system that ingests real-time posts from X (Twitter), performs sentiment analysis using fine-tuned transformer models, and provides predictions via a REST API. The system includes automated model retraining, experiment tracking, and comprehensive monitoring.

### Key Characteristics

- **Real-time Processing**: Streaming data ingestion with X API v2 Filtered Stream
- **Modern Python**: Python 3.12 with type hints, async/await, and modern patterns
- **MLOps Best Practices**: Experiment tracking, model versioning, automated retraining
- **Production-Ready**: Containerized, monitored, and horizontally scalable
- **Cloud-Native**: Designed for deployment on AWS, GCP, or Azure

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Client Applications                            │
│                      (Web, Mobile, CLI, Services)                        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ HTTPS
                                 ▼
                    ┌────────────────────────┐
                    │   Load Balancer / CDN   │
                    │    (Nginx/Cloudflare)   │
                    └────────────┬────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                            API Layer (FastAPI)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Sentiment   │  │   Health     │  │  Retraining  │  │  Monitoring  │  │
│  │  Prediction  │  │   Checks     │  │   Trigger    │  │   Metrics    │  │
│  │  Endpoints   │  │              │  │              │  │  (Prometheus)│  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
└──────────┬───────────────────┬─────────────────┬─────────────────┬────────┘
           │                   │                 │                 │
           ├───────────────────┴─────────────────┴─────────────────┤
           │                                                         │
           ▼                                                         ▼
┌─────────────────────┐                                  ┌──────────────────┐
│   Trained Model     │                                  │  Prometheus      │
│   (Transformers)    │                                  │  (Metrics)       │
│  - DistilBERT       │                                  └──────────────────┘
│  - Tokenizer        │                                           │
└─────────────────────┘                                           ▼
           │                                              ┌──────────────────┐
           │                                              │     Grafana      │
           │                                              │   (Dashboard)    │
           │                                              └──────────────────┘
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Persistence Layer                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────┐   │
│  │   PostgreSQL     │  │      Redis       │  │    Artifact Store   │   │
│  │                  │  │                  │  │   (S3/GCS/Azure)    │   │
│  │ - Posts          │  │ - Cache          │  │                     │   │
│  │ - Predictions    │  │ - Celery Queue   │  │ - Model Files       │   │
│  │ - Models         │  │ - Rate Limiting  │  │ - Checkpoints       │   │
│  │ - Metrics        │  │ - Sessions       │  │ - Datasets          │   │
│  └────────┬─────────┘  └────────┬─────────┘  └─────────────────────┘   │
└───────────┼──────────────────────┼──────────────────────────────────────┘
            │                      │
            ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Background Processing Layer                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                       Celery Workers                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │   │
│  │  │  Training    │  │  Retraining  │  │   Data Processing    │   │   │
│  │  │   Tasks      │  │   Pipeline   │  │   & Aggregation      │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│           │                      │                      │                │
│           └──────────────────────┴──────────────────────┘                │
│                                  │                                       │
└──────────────────────────────────┼───────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Experiment Tracking & Monitoring                    │
│  ┌──────────────────┐                     ┌──────────────────┐          │
│  │     MLflow       │                     │  Weights & Biases│          │
│  │                  │                     │    (Optional)     │          │
│  │ - Experiments    │                     │                  │          │
│  │ - Runs           │                     │ - Experiments    │          │
│  │ - Models         │                     │ - Artifacts      │          │
│  │ - Artifacts      │                     │ - Collaboration  │          │
│  │ - Model Registry │                     │                  │          │
│  └──────────────────┘                     └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
                                   ▲
                                   │
                                   │ Training Metrics
                                   │
┌──────────────────────────────────┴───────────────────────────────────────┐
│                        Data Ingestion Layer                               │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │              X API v2 Filtered Stream Client                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │  │
│  │  │  Stream      │  │  Filter      │  │   Data Validation &     │ │  │
│  │  │  Manager     │  │  Rules       │  │   Storage               │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  │ Bearer Token Auth
                                  ▼
                         ┌────────────────────┐
                         │   X API v2         │
                         │   (Twitter API)    │
                         └────────────────────┘
```

## Component Architecture

### 1. API Layer (FastAPI)

**Purpose**: Serves as the main interface for external clients to interact with the sentiment analysis system.

**Key Components**:
- **Sentiment Prediction Endpoints**: Single and batch prediction capabilities
- **Model Management**: Model info, version history, and retraining triggers
- **Health Checks**: System health monitoring and readiness probes
- **Metrics Exposure**: Prometheus metrics endpoint for observability

**Technology Choices**:
- **FastAPI**: Modern, fast async web framework with automatic OpenAPI documentation
- **Uvicorn**: ASGI server with excellent performance (uses uvloop on Unix)
- **Pydantic**: Data validation and settings management with type safety

**Design Patterns**:
- Dependency injection for database sessions and services
- Response models for consistent API contracts
- Middleware for request logging, CORS, and error handling
- Async endpoints for I/O-bound operations

### 2. Data Ingestion Layer

**Purpose**: Streams real-time posts from X API v2 and stores them for analysis.

**Key Components**:
- **XStreamListener**: Handles incoming posts from X API v2 filtered stream
- **XStreamManager**: Manages stream lifecycle, rules, and error recovery
- **Data Validator**: Validates and normalizes incoming data

**Technology Choices**:
- **Tweepy 4.14.0**: Official Python library for X API v2
- **OAuth 2.0 Bearer Token**: App-Only authentication for filtered stream access

**Design Decisions**:
- Resilient connection handling with automatic reconnection
- Rate limit awareness and exponential backoff
- Separate storage path for raw and processed data
- Asynchronous database writes to avoid blocking the stream

### 3. Text Preprocessing

**Purpose**: Transforms raw text into clean, normalized features suitable for model training and inference.

**Key Components**:
- **TextProcessor**: Main preprocessing pipeline
- **Feature Extractor**: Extracts meaningful features from text

**Preprocessing Steps**:
1. Emoji handling (convert to text descriptions)
2. URL removal
3. Mention and hashtag normalization
4. Lowercasing and tokenization
5. Stop word removal (configurable)
6. Lemmatization

**Technology Choices**:
- **NLTK**: Tokenization, stop words, WordNet
- **spaCy**: Advanced NLP features (optional)
- **emoji**: Emoji to text conversion

### 4. Model Training & Versioning

**Purpose**: Trains transformer-based sentiment analysis models with experiment tracking and versioning.

**Key Components**:
- **SentimentModelTrainer**: Orchestrates training pipeline
- **MLflowTracker**: Logs experiments to MLflow
- **WandBTracker**: Logs experiments to Weights & Biases (optional)
- **CombinedTracker**: Unified interface for multiple tracking systems

**Technology Choices**:
- **Transformers 4.46.0**: HuggingFace library for transformer models
- **PyTorch 2.5.1**: Deep learning framework
- **MLflow 2.18.0**: Experiment tracking and model registry
- **Weights & Biases 0.18.7**: Advanced experiment tracking (optional)

**Model Architecture**:
- Base model: DistilBERT (distilbert-base-uncased)
- Fine-tuned for 3-class sentiment (positive, neutral, negative)
- Configurable hyperparameters via environment variables

**Training Pipeline**:
1. Data preparation and train/val/test split
2. Tokenization with transformer tokenizer
3. Training with HuggingFace Trainer
4. Evaluation on validation set
5. Model saving and versioning
6. Experiment logging to MLflow/W&B

### 5. Automated Retraining Pipeline

**Purpose**: Monitors model performance and triggers retraining when needed.

**Key Components**:
- **RetrainingPipeline**: Orchestrates retraining logic
- **Data Quality Monitor**: Tracks data distribution and drift
- **Performance Monitor**: Tracks model confidence and accuracy

**Retraining Triggers**:
1. **Data Threshold**: Accumulation of N new samples (default: 1000)
2. **Time-Based**: Hours since last training (default: 24 hours)
3. **Performance Degradation**: Model confidence below threshold
4. **Data Drift**: Distribution shift detection

**Technology Choices**:
- **Celery**: Distributed task queue for background processing
- **Redis**: Message broker and result backend

### 6. Persistence Layer

**Databases**:

**PostgreSQL**:
- Primary data store for posts, predictions, and models
- ACID compliance for data integrity
- Full-text search capabilities
- Connection pooling with SQLAlchemy

**Redis**:
- Caching layer for frequent queries
- Celery message broker
- Rate limiting store
- Session management

**Artifact Storage**:
- Model checkpoints and artifacts
- Training datasets
- Supports S3, GCS, Azure Blob Storage, or local filesystem

**Technology Choices**:
- **SQLAlchemy 2.0.36**: ORM and database toolkit
- **Alembic 1.14.0**: Database migrations
- **psycopg2-binary 2.9.10**: PostgreSQL adapter
- **redis 5.2.0**: Redis client

### 7. Monitoring & Observability

**Purpose**: Provides comprehensive visibility into system health and performance.

**Components**:

**Metrics (Prometheus)**:
- API request rates and latency
- Prediction confidence distribution
- Model inference time
- Background task queue depth
- Database query performance

**Experiment Tracking (MLflow)**:
- Training metrics (loss, accuracy, F1-score)
- Model hyperparameters
- Artifact versioning
- Model registry

**Logging**:
- Structured JSON logging
- Log aggregation ready (ELK, Datadog, CloudWatch)
- Correlation IDs for request tracing

**Technology Choices**:
- **prometheus-client 0.21.0**: Metrics instrumentation
- **Grafana**: Visualization and dashboards
- **MLflow UI**: Experiment exploration

### 8. Background Processing (Celery)

**Purpose**: Handles long-running tasks asynchronously to avoid blocking API requests.

**Task Types**:
- Model training and retraining
- Batch prediction processing
- Data aggregation and reporting
- Scheduled maintenance tasks

**Technology Choices**:
- **Celery 5.4.0**: Distributed task queue
- **Redis**: Message broker (alternative: RabbitMQ)

**Configuration**:
- Worker pools for different task priorities
- Task routing for resource optimization
- Result expiration for cleanup
- Rate limiting per task type

## Data Flow

### 1. Real-Time Prediction Flow

```
Client Request
    │
    ▼
[FastAPI Endpoint]
    │
    ├──> [Cache Check (Redis)]
    │         │
    │         ├─ Cache Hit → Return Result
    │         │
    │         └─ Cache Miss ──┐
    │                         │
    └─────────────────────────┘
                │
                ▼
        [Text Preprocessing]
                │
                ▼
        [Model Inference]
                │
                ▼
        [Store Prediction (PostgreSQL)]
                │
                ▼
        [Update Cache (Redis)]
                │
                ▼
        [Log Metrics (Prometheus)]
                │
                ▼
        Response to Client
```

### 2. Training/Retraining Flow

```
Manual Trigger / Scheduled Task / Automatic Trigger
                │
                ▼
    [Celery Task Queued]
                │
                ▼
    [Fetch Training Data from PostgreSQL]
                │
                ▼
    [Data Preprocessing & Splitting]
                │
                ▼
    [Initialize MLflow Run]
                │
                ▼
    [Model Training Loop]
                │
                ├──> [Log Metrics to MLflow]
                │
                └──> [Log Metrics to W&B (optional)]
                │
                ▼
    [Model Evaluation]
                │
                ▼
    [Save Model & Artifacts]
                │
                ├──> [MLflow Model Registry]
                │
                └──> [Artifact Storage (S3/GCS)]
                │
                ▼
    [Update Model Version in Database]
                │
                ▼
    [Reload Model in API (Hot Swap)]
                │
                ▼
    [Send Notification / Update Status]
```

### 3. Streaming Ingestion Flow

```
X API v2 Filtered Stream
        │
        ▼
[XStreamListener receives post]
        │
        ▼
[Parse & Validate Data]
        │
        ▼
[Store Raw Post (PostgreSQL)]
        │
        ▼
[Trigger Preprocessing (Optional)]
        │
        ▼
[Check Retraining Triggers]
        │
        ├─ Threshold Met ──> [Queue Retraining Task]
        │
        └─ Not Met ───────> Continue Streaming
```

## Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Language | Python | 3.12+ | Modern, type-safe, async-capable |
| Web Framework | FastAPI | 0.115.5 | High-performance async API |
| Server | Uvicorn | 0.32.1 | ASGI server with uvloop |
| ML Framework | PyTorch | 2.5.1 | Deep learning training |
| Transformers | HuggingFace Transformers | 4.46.0 | Pre-trained models |
| Database | PostgreSQL | 15+ | Relational data store |
| Cache | Redis | 7+ | In-memory data store |
| Task Queue | Celery | 5.4.0 | Distributed task processing |
| Experiment Tracking | MLflow | 2.18.0 | ML lifecycle management |
| Metrics | Prometheus | Latest | Monitoring and alerting |
| Containerization | Docker | Latest | Application packaging |

### Data & ML Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| scikit-learn | 1.5.2 | ML utilities and metrics |
| pandas | 2.2.3 | Data manipulation |
| numpy | 1.26.4 | Numerical computing |
| nltk | 3.9.1 | NLP preprocessing |
| spacy | 3.8.2 | Advanced NLP |
| emoji | 2.14.0 | Emoji handling |

### API & Data Validation

| Library | Version | Purpose |
|---------|---------|---------|
| pydantic | 2.10.3 | Data validation |
| pydantic-settings | 2.6.1 | Settings management |
| python-dotenv | 1.0.1 | Environment variables |
| httpx | 0.28.1 | HTTP client |

## Design Decisions

### 1. Python 3.12 with Modern Type Hints

**Decision**: Use Python 3.12 with full type hint coverage using PEP 604 union syntax (`|`).

**Rationale**:
- Improved code maintainability and readability
- Better IDE support and autocomplete
- Early error detection with static type checkers
- Self-documenting code

**Trade-offs**:
- Requires Python 3.12+ (not backward compatible with older versions)
- Initial development overhead for adding type hints

### 2. Async FastAPI with Uvicorn

**Decision**: Use async/await patterns with FastAPI and Uvicorn ASGI server.

**Rationale**:
- Non-blocking I/O for database and external API calls
- Better concurrency and resource utilization
- Native support for WebSockets (future feature)
- Excellent performance benchmarks

**Trade-offs**:
- More complex than synchronous code
- Requires careful handling of blocking operations

### 3. DistilBERT as Base Model

**Decision**: Use DistilBERT instead of BERT or larger models.

**Rationale**:
- 40% smaller than BERT with 97% of performance
- Faster inference (2x speedup)
- Lower memory footprint
- Good balance of accuracy and speed for production

**Trade-offs**:
- Slightly lower accuracy than full BERT
- May need fine-tuning for domain-specific tasks

### 4. MLflow for Experiment Tracking

**Decision**: Use MLflow as primary experiment tracking system with optional W&B integration.

**Rationale**:
- Open-source and self-hostable
- Model registry with versioning
- Framework-agnostic
- Good UI for experiment comparison
- Easy integration with deployment

**Trade-offs**:
- Requires separate server deployment
- UI less polished than W&B

### 5. PostgreSQL as Primary Database

**Decision**: Use PostgreSQL for all structured data storage.

**Rationale**:
- ACID compliance for data integrity
- Full-text search capabilities
- JSON support for flexible schemas
- Mature ecosystem and tooling
- Excellent performance for read-heavy workloads

**Trade-offs**:
- Requires operational overhead (backups, scaling)
- Vertical scaling limits (mitigated by read replicas)

### 6. Redis for Caching and Message Broker

**Decision**: Use Redis for both caching and Celery message broker.

**Rationale**:
- Single dependency for multiple use cases
- Excellent performance for in-memory operations
- Simple key-value model
- Built-in expiration for automatic cleanup
- Supports pub/sub for real-time features

**Trade-offs**:
- Data persistence options limited
- Memory constraints require careful sizing

### 7. Celery for Background Tasks

**Decision**: Use Celery for all long-running background tasks.

**Rationale**:
- Battle-tested distributed task queue
- Flexible task routing and prioritization
- Built-in retry mechanisms
- Monitoring and management tools (Flower)
- Supports multiple backends

**Trade-offs**:
- Requires message broker (Redis/RabbitMQ)
- Can be complex to debug

### 8. Docker for Containerization

**Decision**: Fully containerized application with Docker Compose for local development.

**Rationale**:
- Consistent environments across dev/staging/prod
- Easy dependency management
- Supports orchestration (Kubernetes, ECS)
- Simplified deployment process
- Isolation and security

**Trade-offs**:
- Additional layer of complexity
- Resource overhead

## Scalability Considerations

### Horizontal Scaling Strategies

#### API Layer
- **Stateless Design**: No server-side session storage (use Redis or JWT)
- **Load Balancing**: Deploy multiple API instances behind nginx/ALB
- **Auto-scaling**: Scale based on CPU/memory or request queue depth
- **Health Checks**: Implement liveness and readiness probes

#### Data Ingestion
- **Multiple Stream Consumers**: Deploy multiple stream listeners with different filters
- **Partitioning**: Distribute stream processing across multiple workers
- **Buffering**: Use message queues to handle burst traffic

#### Background Processing
- **Worker Pools**: Separate workers for different task types (training vs. prediction)
- **Task Priority**: High-priority tasks (API-triggered) vs. low-priority (scheduled)
- **Resource Isolation**: GPU workers for training, CPU workers for data processing

### Vertical Scaling Options

- **Database**: Read replicas for query distribution
- **Model Inference**: GPU acceleration for faster predictions
- **Caching**: Larger Redis instance for more cached results

### Performance Optimizations

1. **Database Query Optimization**
   - Indexing on frequently queried columns
   - Connection pooling with SQLAlchemy
   - Query result caching

2. **Model Inference**
   - Batch prediction for efficiency
   - Model quantization for faster inference
   - TorchScript compilation for production

3. **Caching Strategy**
   - Cache prediction results for common inputs
   - Cache-aside pattern with TTL
   - Warm-up caches on deployment

4. **Asynchronous Operations**
   - Non-blocking database calls
   - Concurrent prediction requests
   - Background metric logging

## Security Architecture

### Authentication & Authorization

- **API Key Authentication**: For service-to-service communication
- **JWT Tokens**: For user authentication (if needed)
- **Rate Limiting**: Per-client request throttling with Redis

### Data Security

- **Encryption at Rest**: Database encryption with PostgreSQL
- **Encryption in Transit**: HTTPS/TLS for all external communication
- **Secrets Management**: Environment variables or HashiCorp Vault
- **Data Sanitization**: Input validation with Pydantic

### Network Security

- **VPC/Private Network**: Isolate backend services
- **Security Groups**: Restrict access to databases and internal services
- **API Gateway**: Single entry point with WAF protection

### Compliance Considerations

- **Data Retention**: Configurable data retention policies
- **Audit Logging**: Track all data access and modifications
- **GDPR Compliance**: User data deletion capabilities
- **PII Handling**: Minimize collection and storage of personal information

## Monitoring & Alerting Strategy

### Key Metrics to Monitor

**Application Health**:
- API response time (p50, p95, p99)
- Error rates (4xx, 5xx)
- Request throughput
- Active connections

**Model Performance**:
- Prediction confidence distribution
- Model inference latency
- Batch processing time
- Training job duration

**Infrastructure**:
- CPU and memory utilization
- Database connection pool usage
- Celery queue depth
- Redis memory usage

### Alert Thresholds

- API latency > 500ms (p95) → Warning
- Error rate > 1% → Critical
- Model confidence < 0.6 → Warning
- Celery queue depth > 1000 → Warning
- Database connection pool > 80% → Warning

## Future Enhancements

### Short-term (1-3 months)
- [ ] Implement model A/B testing framework
- [ ] Add GraphQL API option
- [ ] Enhanced data drift detection
- [ ] Multi-language support

### Medium-term (3-6 months)
- [ ] Active learning pipeline
- [ ] Model interpretability (SHAP/LIME)
- [ ] Real-time prediction dashboard
- [ ] Advanced data augmentation

### Long-term (6+ months)
- [ ] Multi-model ensemble predictions
- [ ] Edge deployment for inference
- [ ] Federated learning support
- [ ] Custom sentiment categories per client

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [X API v2 Documentation](https://developer.x.com/en/docs/x-api)
- [The Twelve-Factor App](https://12factor.net/)
