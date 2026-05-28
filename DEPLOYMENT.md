# BlitzMate Deployment Guide

This guide provides instructions for deploying the BlitzMate backend chess engine API and the Next.js frontend web interface.

---

## 1. Backend Deployment (Hugging Face Spaces)

The backend runs inside a Docker container. Hugging Face Spaces provides free CPU-based Docker hosting, making it ideal for the BlitzMate stateless API.

### Steps to Deploy

1. **Create a Hugging Face Account**: If you don't have one, sign up at [huggingface.co](https://huggingface.co/).
2. **Create a New Space**:
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - Give your Space a name (e.g., `blitzmate-engine`).
   - Select **Docker** as the SDK.
   - Choose the **Blank** template (or select Python).
   - Set the space visibility to **Public** (required so the frontend can reach the API).
   - Choose the free **CPU Basic** hardware.
3. **Upload the Code**:
   - Clone your Hugging Face Space repository locally, or upload the files directly via the HF web interface.
   - You only need to push the following files/directories from the root:
     - `engine/`
     - `server/`
     - `Dockerfile`
     - `.dockerignore`
   - *Note*: Opening books and Syzygy tablebases are ignored via `.dockerignore` to keep the Docker image small and build times fast.
4. **Accessing your API**:
   - Once the build completes, the Space will be running at:
     `https://<username>-<space-name>.hf.space`
   - Verify it is running by checking the health endpoint:
     `https://<username>-<space-name>.hf.space/health`

---

## 2. Frontend Deployment (Vercel)

The Next.js frontend is fully compatible with Vercel's edge-optimized hosting.

### Steps to Deploy

1. **Create a Vercel Account**: Sign up at [vercel.com](https://vercel.com).
2. **Import your Repository**:
   - Create a new project on Vercel and link your GitHub repository.
3. **Configure Project Settings**:
   - **Framework Preset**: Next.js
   - **Root Directory**: `web` (since the Next.js app is located in the `web/` subfolder).
4. **Add Environment Variables**:
   - Add the following environment variable to the Vercel build settings:
     - **Key**: `NEXT_PUBLIC_ENGINE_API_URL`
     - **Value**: The URL of your Hugging Face Space (e.g., `https://username-space-name.hf.space`)
5. **Deploy**:
   - Click **Deploy**. Vercel will automatically build the Next.js static files and deploy the application.
