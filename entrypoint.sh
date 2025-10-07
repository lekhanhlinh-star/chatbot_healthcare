#!/bin/bash
# Start Ollama server in background
ollama serve &
OLLAMA_PID=$!

# Wait until Ollama server is ready
echo "Waiting for Ollama server to start..."
while ! curl -s http://127.0.0.1:11434/api/tags >/dev/null; do
    sleep 2
done
echo "Ollama server is ready!"

# Models are already pulled during build, so we can start the app immediately
echo "Starting FastAPI application with uvicorn..."
uvicorn app:app --host 0.0.0.0 --port ${PORT:-80}