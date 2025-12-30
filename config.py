from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Configuration for Reddit Sentiment Analysis Pipeline
    
    Reddit API Requirements:
    - Client ID and Client Secret - REQUIRED
    - Completely FREE with no credit card required
    - Create app at: https://www.reddit.com/prefs/apps
    - Select "script" type for personal use
    - Rate limit: 60 requests/minute (very generous)
    """
    
    # Reddit API Credentials - REQUIRED (all free)
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str = "sentiment-analysis-bot/1.0"
    
    wandb_api_key: str | None = None
    wandb_project: str = "reddit-sentiment-analysis"
    
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "sentiment-analysis"
    
    database_url: str = "postgresql://user:password@localhost:5432/reddit_sentiment"
    redis_url: str = "redis://localhost:6379/0"
    
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    min_samples_for_retrain: int = 1000
    retrain_interval_hours: int = 24
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
