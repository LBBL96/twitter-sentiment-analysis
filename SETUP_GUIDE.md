# Setup Guide

This guide will walk you through setting up the Reddit Sentiment Analysis Pipeline from scratch.

## Prerequisites Installation

### 1. Install Python 3.10+
```bash
python --version  # Should be 3.10 or higher
```

### 2. Install PostgreSQL
```bash
# macOS
brew install postgresql@15
brew services start postgresql@15

# Ubuntu/Debian
sudo apt-get install postgresql-15
sudo systemctl start postgresql

# Create database
createdb reddit_sentiment
```

### 3. Install Redis
```bash
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis
```

### 4. Install Docker (Optional but Recommended)
Download from: https://www.docker.com/products/docker-desktop

## Project Setup

### Step 1: Clone and Create Virtual Environment
```bash
git clone <your-repo-url>
cd reddit-sentiment-analysis

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 3: Configure Environment
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
# Reddit API Credentials (Get from https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=sentiment-bot/1.0

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/reddit_sentiment

# Redis
REDIS_URL=redis://localhost:6379/0

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Weights & Biases (Optional)
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=reddit-sentiment-analysis
```

### Step 4: Initialize Database
```bash
python scripts/init_db.py
```

## Getting Reddit API Credentials

1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Select "script" type for personal use
4. Fill in name and redirect URI (use http://localhost:8080 for local)
5. Copy the client ID and client secret to `.env` file
6. **Completely FREE - no credit card required!**

## Running the Application

### Option 1: Using Docker Compose (Recommended)
```bash
docker-compose up -d
```

This starts all services:
- API: http://localhost:8000
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### Option 2: Running Locally

#### Terminal 1: Start MLflow
```bash
mlflow server \
  --backend-store-uri postgresql://user:password@localhost:5432/mlflow_db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

#### Terminal 2: Start Celery Worker
```bash
celery -A src.tasks.celery_app worker --loglevel=info
```

#### Terminal 3: Start Celery Beat (Scheduler)
```bash
celery -A src.tasks.celery_app beat --loglevel=info
```

#### Terminal 4: Start API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## First Steps After Setup

### 1. Load Sample Training Data
```bash
# Load sentiment140 dataset
python scripts/load_sample_data.py --source sentiment140

# Or load custom samples
python scripts/load_sample_data.py --source custom
```

### 2. Train Initial Model
```bash
python scripts/train_model.py --force
```

### 3. Start Reddit Stream (Optional)
```bash
python scripts/stream_reddit.py --subreddits "technology" "artificial" "python"
```

### 4. Test API
```bash
# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this amazing product!"}'

# Check health
curl http://localhost:8000/health
```

## Monitoring Setup

### Access Grafana
1. Open http://localhost:3000
2. Login: admin/admin
3. Add Prometheus datasource: http://prometheus:9090
4. Import dashboards or create custom ones

### Access MLflow
1. Open http://localhost:5000
2. View experiments and model versions
3. Compare training runs

### Access W&B (if configured)
1. Login to https://wandb.ai
2. View your project dashboard

## Troubleshooting

### Issue: Database Connection Error
```bash
# Check PostgreSQL is running
pg_isready

# Check connection string in .env
# Ensure database exists
psql -l
```

### Issue: Redis Connection Error
```bash
# Check Redis is running
redis-cli ping  # Should return PONG

# Check Redis URL in .env
```

### Issue: Reddit API Rate Limit
- PRAW handles rate limiting automatically (60 requests/minute)
- Add delays between requests if needed
- Reddit API is very generous with free tier

### Issue: Model Loading Error
```bash
# Check model files exist
ls -la trained_models/

# Retrain model
python scripts/train_model.py --force
```

### Issue: Import Errors
```bash
# Ensure you're in virtual environment
which python

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Next Steps

1. **Configure monitoring alerts** in Grafana
2. **Set up GitHub Actions secrets** for CI/CD
3. **Customize model hyperparameters** in `.env`
4. **Add custom training data** to improve accuracy
5. **Deploy to production** using Docker or cloud services

## Production Deployment Checklist

- [ ] Change default passwords (Grafana, database)
- [ ] Enable HTTPS/SSL
- [ ] Set up backup strategy
- [ ] Configure log rotation
- [ ] Set up monitoring alerts
- [ ] Enable authentication on API endpoints
- [ ] Configure CORS properly
- [ ] Set up rate limiting
- [ ] Enable auto-scaling
- [ ] Document disaster recovery procedures

## Support

If you encounter issues:
1. Check logs: `docker-compose logs <service-name>`
2. Review documentation in README.md
3. Check GitHub issues
4. Verify all environment variables are set correctly
