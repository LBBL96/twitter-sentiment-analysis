# Reddit Sentiment Analysis Pipeline

![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.5-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.18.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)

A production-ready, real-time sentiment analysis system for Reddit data with streaming ingestion, transformer-based ML models, automated retraining, and comprehensive MLOps integration. **100% FREE** - no credit card or paid API access required!

## 📚 Documentation

- **[Architecture Overview](ARCHITECTURE.md)** - System design, components, and data flow
- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment strategies and infrastructure
- **[Reddit API Setup](REDDIT_SETUP.md)** - Quick guide to get free Reddit API credentials

## ✨ Features

### Core Capabilities
- **🔄 Real-Time Data Ingestion**: Stream posts and comments using **Reddit API (PRAW)** - completely free!
- **🧹 Advanced Text Preprocessing**: NLP preprocessing with emoji handling, URL removal, and tokenization (NLTK, spaCy)
- **🤖 Transformer Models**: Fine-tune BERT/DistilBERT models with HuggingFace Transformers
- **📊 MLOps Integration**: Full experiment tracking with MLflow and Weights & Biases
- **🔁 Automated Retraining**: Intelligent retraining based on data drift and performance degradation
- **🚀 Production-Ready API**: FastAPI with async support, automatic OpenAPI docs, and Prometheus metrics
- **📦 Containerized**: Docker and Docker Compose for consistent deployments

### Technical Highlights
- **Modern Python 3.12**: Type hints with PEP 604 syntax, async/await patterns
- **Horizontal Scaling**: Stateless API design with Redis caching and Celery workers
- **Comprehensive Monitoring**: Prometheus metrics, structured logging, health checks
- **Database Migrations**: Alembic for version-controlled schema changes
- **Security Best Practices**: Input validation, secrets management, TLS/SSL ready

## Architecture

```
├── src/
│   ├── data_ingestion/       # Reddit streaming and data collection
│   ├── preprocessing/         # Text preprocessing and feature engineering
│   ├── model_training/        # Model training, versioning, and retraining
│   ├── api/                   # FastAPI application
│   ├── tasks/                 # Celery background tasks
│   └── database/              # Database models and connections
├── tests/                     # Unit and integration tests
├── scripts/                   # Utility scripts
├── .github/workflows/         # CI/CD pipelines
└── docker-compose.yml         # Docker orchestration
```

## 📋 Prerequisites

### Required
- **Python 3.12+** (uses modern type hints and features)
- **Docker & Docker Compose** (recommended for easy setup)
- **Reddit API Access** (100% FREE):
  - Reddit account (free)
  - Create app at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
  - Get Client ID and Client Secret (takes 2 minutes)
  - **No credit card required!**
  - Rate limit: 60 requests/minute (very generous)

### Optional (for manual installation)
- PostgreSQL 15+
- Redis 7+
- Weights & Biases account (enhanced experiment tracking)

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone and configure
git clone https://github.com/yourusername/reddit-sentiment-analysis.git
cd reddit-sentiment-analysis
cp .env.example .env
# Edit .env with your Reddit API credentials (free - see setup below)

# 2. Start all services
docker compose up -d

# 3. Access the application
open http://localhost:8000/docs
```

That's it! All services (API, Database, Redis, MLflow) are now running.

### Option 2: Manual Installation

#### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd reddit-sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your credentials
```

Required environment variables:
- `REDDIT_CLIENT_ID` - **REQUIRED** (free from reddit.com/prefs/apps)
- `REDDIT_CLIENT_SECRET` - **REQUIRED** (free from reddit.com/prefs/apps)
- `REDDIT_USER_AGENT` - Your app identifier (e.g., "sentiment-bot/1.0")
- `DATABASE_URL` (PostgreSQL connection string)
- `REDIS_URL` (Redis connection string)
- `MLFLOW_TRACKING_URI` (MLflow server URL)
- `WANDB_API_KEY` (optional, for W&B tracking)

### 3. Initialize Database

```bash
python scripts/init_db.py
```

### 4. Start Services with Docker Compose

```bash
docker compose up -d
```

This starts:
- FastAPI application (port 8000)
- PostgreSQL database (port 5432)
- Redis (port 6379)
- MLflow server (port 5000)
- Celery workers for background tasks
- Prometheus (port 9090)
- Grafana (port 3000)

### 5. Access the Application

- **API Documentation**: http://localhost:8000/docs (Interactive Swagger UI)
- **API Health Check**: http://localhost:8000/health
- **MLflow UI**: http://localhost:5000 (Experiment tracking)
- **Prometheus**: http://localhost:9090 (Metrics)
- **Grafana**: http://localhost:3000 (Dashboards - admin/admin)

## 🎯 Demo Credentials

For quick testing without Reddit API access:

```bash
# Use the demo endpoint with sample data
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this amazing product! Best purchase ever!"}'

# Expected response:
{
  "sentiment": "positive",
  "confidence": 0.94,
  "scores": {
    "positive": 0.94,
    "neutral": 0.04,
    "negative": 0.02
  }
}
```

See the [API Documentation](http://localhost:8000/docs) for all available endpoints.

## Usage

### Stream Reddit Data

```python
from src.data_ingestion.reddit_streamer import RedditStreamManager

manager = RedditStreamManager()
subreddits = ['technology', 'MachineLearning', 'artificial']
manager.stream_subreddit_submissions(subreddits, save_to_db=True)
```

### Train a Model

```python
from src.model_training.trainer import SentimentModelTrainer

trainer = SentimentModelTrainer()
train_dataset, val_dataset, test_dataset = trainer.prepare_data(texts, labels)
trainer.train(train_dataset, val_dataset)
model_path = trainer.save_model('./trained_models', version='v1.0.0')
```

### Make Predictions via API

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this amazing product!"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great service!", "Terrible experience", "It was okay"]}'
```

### Trigger Retraining

```bash
curl -X POST "http://localhost:8000/retrain" \
  -H "Content-Type: application/json" \
  -d '{"force": false}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check and model status |
| `/predict` | POST | Single text sentiment prediction |
| `/predict/batch` | POST | Batch sentiment predictions |
| `/retrain` | POST | Trigger model retraining |
| `/model/info` | GET | Current and recent model information |
| `/data/quality` | GET | Data quality and drift monitoring |
| `/metrics` | GET | Prometheus metrics |

## Model Training & Versioning

The system uses MLflow and Weights & Biases for experiment tracking:

1. **Automatic Logging**: All training runs are logged with hyperparameters and metrics
2. **Model Registry**: Models are versioned and stored with metadata
3. **Experiment Comparison**: Compare different model versions in MLflow/W&B UI
4. **Artifact Storage**: Model checkpoints and artifacts are stored and versioned

## Automated Retraining

The retraining pipeline automatically triggers when:

1. **Data Threshold**: Accumulation of N new unlabeled samples (configurable)
2. **Time-Based**: After X hours since last training (configurable)
3. **Performance Degradation**: When model confidence drops below threshold
4. **Data Drift**: When significant distribution shift is detected

Configure thresholds in `.env`:
```
MIN_SAMPLES_FOR_RETRAIN=1000
RETRAIN_INTERVAL_HOURS=24
```

## Monitoring & Observability

### Prometheus Metrics

- `sentiment_predictions_total`: Total number of predictions
- `sentiment_prediction_duration_seconds`: Prediction latency
- `sentiment_prediction_confidence`: Model confidence distribution

### MLflow Tracking

- Training metrics (loss, accuracy, F1-score)
- Model parameters and hyperparameters
- Model artifacts and checkpoints

### Data Quality Monitoring

- Sentiment distribution tracking
- Prediction confidence monitoring
- Data drift detection
- Model performance degradation alerts

## CI/CD Pipeline

GitHub Actions workflows:

1. **CI Pipeline** (`.github/workflows/ci.yml`):
   - Code linting (flake8, black)
   - Type checking (mypy)
   - Unit tests with coverage
   - Security scanning (Trivy)
   - Docker image build

2. **CD Pipeline** (`.github/workflows/cd.yml`):
   - Docker image build and push
   - Automated deployment
   - Model retraining trigger

### Required GitHub Secrets

- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`
- `DOCKER_USERNAME`, `DOCKER_PASSWORD`
- `DEPLOY_KEY`, `DEPLOY_HOST`
- `MLFLOW_TRACKING_URI`, `WANDB_API_KEY`
- `DATABASE_URL`

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py
```

## Development

### Code Formatting

```bash
# Format code
black src/

# Check formatting
black --check src/

# Lint code
flake8 src/
```

### Type Checking

```bash
mypy src/ --ignore-missing-imports
```

## Deployment

### Production Deployment

1. **Update environment variables** for production
2. **Build Docker image**:
   ```bash
   docker build -t sentiment-analysis:latest .
   ```
3. **Deploy using docker-compose** or orchestration tool (Kubernetes, ECS, etc.)
4. **Set up monitoring and alerting**
5. **Configure backup and disaster recovery**

### Scaling Considerations

- Use Redis for caching and rate limiting
- Deploy multiple API instances behind a load balancer
- Use Celery for distributed task processing
- Consider using GPU instances for model training
- Implement horizontal scaling for API workers

## Configuration

Key configuration options in `config.py`:

- `MODEL_NAME`: Base transformer model (default: distilbert-base-uncased)
- `MAX_LENGTH`: Maximum sequence length for tokenization
- `BATCH_SIZE`: Training and inference batch size
- `LEARNING_RATE`: Model training learning rate
- `NUM_EPOCHS`: Number of training epochs
- `MIN_SAMPLES_FOR_RETRAIN`: Minimum samples to trigger retraining
- `RETRAIN_INTERVAL_HOURS`: Hours between automatic retraining

## Troubleshooting

### Common Issues

1. **Reddit API Rate Limits**: PRAW handles rate limiting automatically (60 req/min)
2. **Memory Issues**: Reduce batch size or use gradient accumulation
3. **Database Connection Errors**: Check PostgreSQL is running and credentials are correct
4. **Model Loading Errors**: Ensure model files exist and are not corrupted

### Logs

```bash
# API logs
docker-compose logs api

# Celery worker logs
docker-compose logs celery_worker

# MLflow logs
docker-compose logs mlflow
```


## Support

For issues and questions:
- Create an issue on GitHub
- Check existing documentation
- Review MLflow and W&B dashboards for model insights

## 🗺️ Roadmap

### Short-term
- [ ] Model A/B testing framework
- [ ] GraphQL API option
- [ ] Enhanced data drift detection
- [ ] Multi-language support

### Medium-term
- [ ] Active learning pipeline
- [ ] Model interpretability (SHAP, LIME)
- [ ] Real-time prediction dashboard
- [ ] Advanced data augmentation

### Long-term
- [ ] Multi-model ensemble predictions
- [ ] Edge deployment for inference
- [ ] Federated learning support
- [ ] Custom sentiment categories per client

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure all tests pass and code is formatted (`black src/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [HuggingFace Transformers](https://huggingface.co/transformers/) for transformer models
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [MLflow](https://mlflow.org/) for experiment tracking
- [PRAW](https://praw.readthedocs.io/) for the Reddit API wrapper
- [Reddit](https://www.reddit.com/) for free API access
