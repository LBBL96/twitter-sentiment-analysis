# Reddit API Setup Guide

Getting Reddit API credentials is **100% FREE** and takes only 2 minutes!

## Step-by-Step Instructions

### 1. Create a Reddit Account
If you don't have one already, sign up at [reddit.com](https://www.reddit.com)

### 2. Create a Reddit App

1. Go to [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Scroll to the bottom and click **"Create App"** or **"Create Another App"**
3. Fill in the form:
   - **name**: Choose any name (e.g., "sentiment-analysis-bot")
   - **App type**: Select **"script"** (for personal use)
   - **description**: Optional (e.g., "Sentiment analysis research")
   - **about url**: Leave blank or add your GitHub repo
   - **redirect uri**: Use `http://localhost:8080` (required but not used for scripts)
4. Click **"Create app"**

### 3. Get Your Credentials

After creating the app, you'll see:

```
personal use script
[CLIENT_ID]          ← This is your CLIENT_ID (under "personal use script")

secret
[CLIENT_SECRET]      ← This is your CLIENT_SECRET
```

### 4. Update Your `.env` File

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=sentiment-analysis-bot/1.0
```

**Important**: The `REDDIT_USER_AGENT` should be descriptive and unique. Format: `<platform>:<app ID>:<version> (by u/<Reddit username>)`

Example: `linux:sentiment-bot:v1.0.0 (by u/yourusername)`

### 5. Test Your Connection

```python
import praw

reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="sentiment-analysis-bot/1.0"
)

# Test by fetching a subreddit
subreddit = reddit.subreddit("python")
print(f"Connected! Subreddit: {subreddit.display_name}")
```

## Rate Limits

Reddit API has generous rate limits:
- **60 requests per minute** for OAuth apps
- PRAW (Python Reddit API Wrapper) handles rate limiting automatically
- No monthly limits or costs
- No credit card required

## Best Practices

1. **User Agent**: Always use a descriptive user agent
2. **Rate Limiting**: PRAW handles this automatically - don't worry about it
3. **Respect Reddit's Rules**: Don't spam or abuse the API
4. **Read-Only Access**: Script apps are read-only (perfect for data collection)

## Recommended Subreddits for Sentiment Analysis

### Technology & News
- `technology`, `gadgets`, `futurology`
- `news`, `worldnews`, `politics`

### Products & Reviews
- `BuyItForLife`, `ProductPorn`, `shutupandtakemymoney`
- `Android`, `apple`, `gaming`

### Discussion & Opinion
- `AskReddit`, `unpopularopinion`, `changemyview`
- `TrueOffMyChest`, `rant`

### Specific Topics
- `MachineLearning`, `artificial`, `datascience`
- `stocks`, `investing`, `CryptoCurrency`
- `movies`, `television`, `books`

## Troubleshooting

### "Invalid Credentials" Error
- Double-check your `CLIENT_ID` and `CLIENT_SECRET`
- Make sure there are no extra spaces in your `.env` file
- Verify you selected "script" type when creating the app

### "Too Many Requests" Error
- PRAW handles rate limiting automatically
- If you see this, you're making requests too fast manually
- Add delays between requests if not using PRAW's stream methods

### "Forbidden" or "401" Error
- Check your user agent is set correctly
- Ensure your Reddit app is still active at reddit.com/prefs/apps

## Need Help?

- [PRAW Documentation](https://praw.readthedocs.io/)
- [Reddit API Documentation](https://www.reddit.com/dev/api/)
- [Reddit API Support](https://www.reddit.com/r/redditdev/)
