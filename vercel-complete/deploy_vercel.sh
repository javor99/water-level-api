#!/bin/bash
echo "🚀 Deploying Flask App to Vercel..."
echo "=================================="

# Install Vercel CLI if not already installed
if ! command -v vercel &> /dev/null; then
    echo "📦 Installing Vercel CLI..."
    npm install -g vercel
fi

# Login to Vercel (if not already logged in)
echo "🔐 Logging into Vercel..."
vercel login

# Deploy the app
echo "🚀 Deploying to Vercel..."
vercel --prod

echo "✅ Deployment complete!"
echo "Your app will be available at the URL provided above."
