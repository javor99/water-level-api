# 🌐 Flask Server Deployment Guide

## Option 1: Vercel (Recommended - FREE)

### Quick Deploy:
```bash
# Run the deployment script
./deploy_vercel.sh
```

### Manual Deploy:
```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy
vercel --prod
```

**Benefits:**
- ✅ FREE forever
- ✅ Static URL (your-app.vercel.app)
- ✅ Automatic HTTPS
- ✅ Global CDN
- ✅ Easy GitHub integration
- ✅ Automatic deployments on git push

---

## Option 2: PythonAnywhere (FREE)

### Steps:
1. Go to [pythonanywhere.com](https://pythonanywhere.com)
2. Create a free account
3. Upload your code via zip file
4. Configure WSGI file
5. Your app will be at: `yourusername.pythonanywhere.com`

**Benefits:**
- ✅ FREE tier available
- ✅ Python-specific hosting
- ✅ Easy setup
- ✅ Static URL

---

## Option 3: Railway (FREE)

### Steps:
1. Go to [railway.app](https://railway.app)
2. Connect GitHub account
3. Deploy from repository
4. Get static URL: `your-app.railway.app`

**Benefits:**
- ✅ FREE tier
- ✅ Easy GitHub integration
- ✅ Automatic deployments
- ✅ Database support

---

## Option 4: Render (FREE)

### Steps:
1. Go to [render.com](https://render.com)
2. Connect GitHub
3. Create new Web Service
4. Deploy from repository

**Benefits:**
- ✅ FREE tier
- ✅ Static URL
- ✅ Easy setup
- ✅ Auto-deployments

---

## Option 5: Heroku (PAID - $5/month)

### Steps:
1. Install Heroku CLI
2. Create Procfile
3. Deploy with git

**Benefits:**
- ✅ Reliable hosting
- ✅ Custom domains
- ✅ Add-ons available
- ❌ No longer free

---

## Environment Variables Setup

For any deployment, you'll need to set these environment variables:

```bash
# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
FROM_EMAIL=your-email@gmail.com
FROM_NAME=Water Level Alert System

# Flask Configuration
FLASK_DEBUG=0
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
```

---

## Recommended: Vercel Deployment

**Why Vercel?**
- Completely FREE
- No credit card required
- Static URL that never changes
- Automatic HTTPS
- Global CDN for fast loading
- Easy GitHub integration
- Professional hosting

**Your app will be available at:**
`https://your-project-name.vercel.app`

**To deploy right now:**
```bash
./deploy_vercel.sh
```

