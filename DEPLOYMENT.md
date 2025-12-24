# Deploying Tactical Touchdown to Google Cloud

This guide walks you through deploying the Tactical Touchdown football simulation game to Google Cloud Run with a Cloud SQL database.

## Prerequisites

1. **Google Cloud Account** - Free tier available at [cloud.google.com](https://cloud.google.com)
2. **gcloud CLI** - Install from [cloud.google.com/sdk](https://cloud.google.com/sdk)
3. **GitHub Repository** - Push your code to GitHub

## Step 1: Create a Google Cloud Project

```bash
# Create a new project
gcloud projects create tactical-touchdown --name="Tactical Touchdown"

# Set as active project
gcloud config set project tactical-touchdown

# Enable required APIs
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  sqladmin.googleapis.com \
  containerregistry.googleapis.com
```

## Step 2: Create Cloud SQL Database

```bash
# Create PostgreSQL instance (free tier eligible)
gcloud sql instances create tactical-touchdown-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1

# Create database
gcloud sql databases create tactical_touchdown \
  --instance=tactical-touchdown-db

# Create user and password
gcloud sql users create gameuser \
  --instance=tactical-touchdown-db \
  --password=YOUR_SECURE_PASSWORD_HERE
```

## Step 3: Connect GitHub to Cloud Build

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to **Cloud Build** → **Repositories**
3. Click **Connect Repository**
4. Select **GitHub** and authorize
5. Select your `TacticalTouchdown` repository
6. Click **Connect**

## Step 4: Set Up Cloud Build Trigger

```bash
# Create trigger that deploys on push to main
gcloud builds triggers create github \
  --repo-name=TacticalTouchdown \
  --repo-owner=YOUR_GITHUB_USERNAME \
  --branch-pattern="^main$" \
  --build-config=cloudbuild.yaml
```

## Step 5: Configure Environment Variables

```bash
# Set Cloud SQL connection for Cloud Run
gcloud run services update tactical-touchdown \
  --set-env-vars=DATABASE_URL="postgresql://gameuser:YOUR_PASSWORD@/tactical_touchdown?host=/cloudsql/tactical-touchdown:us-central1:tactical-touchdown-db" \
  --region=us-central1
```

## Step 6: Deploy Manually (First Time)

```bash
# Build and deploy
gcloud builds submit --config=cloudbuild.yaml
```

## Step 7: Access Your App

Once deployed, get your service URL:

```bash
gcloud run services describe tactical-touchdown \
  --region=us-central1 \
  --format='value(status.url)'
```

Your game will be live at something like: `https://tactical-touchdown-xxxxx.a.run.app`

## Automatic Deployments

Now whenever you push to `main` branch on GitHub, Cloud Build will:
1. Build your Docker image
2. Push to Container Registry
3. Deploy to Cloud Run

## Monitoring & Logs

```bash
# View Cloud Run logs
gcloud run logs read tactical-touchdown \
  --region=us-central1 \
  --limit=50

# View Cloud Build logs
gcloud builds log <BUILD_ID>

# Monitor Cloud SQL
gcloud sql instances describe tactical-touchdown-db
```

## Cost Management

**Free tier usage (3-100 users):**
- Cloud Run: 2M requests/month
- Cloud SQL: Shared instance with 10GB storage
- Total: **$0/month**

**Upgrade when needed:**
- Dedicated Cloud SQL instance: ~$10-15/month
- Additional Cloud Run resources: Pay per request (~$0.00002/request)

## Troubleshooting

### Docker build fails
```bash
# Test locally
docker build -t tactical-touchdown .
docker run -p 8080:8080 tactical-touchdown
```

### Cloud SQL connection fails
- Ensure the connection string matches your instance name
- Check Cloud SQL Auth Proxy is enabled
- Verify network permissions in Cloud SQL

### App crashes after deployment
```bash
# Check logs
gcloud run logs read tactical-touchdown --region=us-central1

# Check environment variables
gcloud run services describe tactical-touchdown --region=us-central1
```

## Next Steps

1. **Set up CI/CD** - Auto-deploy on push to main (already configured)
2. **Monitor performance** - Use Cloud Monitoring dashboard
3. **Scale database** - Upgrade to dedicated instance when needed
4. **Add authentication** - Implement user login system
5. **Backup data** - Set up automated Cloud SQL backups

## Support

For Google Cloud issues, check:
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud SQL Documentation](https://cloud.google.com/sql/docs)
- [Cloud Build Documentation](https://cloud.google.com/build/docs)
