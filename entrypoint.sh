#!/bin/sh
# BlitzMate Docker entrypoint – download opening books then start the server.
set -e

echo "[entrypoint] Setting up opening books..."
python setup_assets.py books

echo "[entrypoint] Starting server..."
exec uvicorn server.app.main:app --host 0.0.0.0 --port 7860
