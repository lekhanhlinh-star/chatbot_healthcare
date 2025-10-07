# Chatbot Healthcare v2

Hệ thống chatbot chăm sóc sức khỏe sử dụng RAG (Retrieval-Augmented Generation) với Ollama và HuggingFace models. Backend được xây dựng với **FastAPI** để có hiệu suất cao và API documentation tự động.

## 🚀 Quick Start

### Yêu cầu
- Docker & Docker Compose
- NVIDIA Docker runtime (để sử dụng GPU)
- Ít nhất 16GB RAM
- 50GB disk space (cho models)

### Chạy với Docker Compose

```bash
# Clone repository
git clone <your-repo>
cd chatbot_healthcare_v2

# Build và start services với FastAPI
docker-compose up -d

# Hoặc sử dụng Makefile
make build
make up
```

### Truy cập ứng dụng

- **Web UI**: http://localhost:80
- **Ollama API**: http://localhost:11434
- **Health check**: http://localhost:80/ping

## 📁 Cấu trúc dự án

```
chatbot_healthcare_v2/
├── app.py                     # Flask application
├── rag_inference.py           # RAG logic và model
├── Dockerfile                 # Container definition
├── docker-compose.yml        # Service orchestration
├── docker-compose.override.yml # Development overrides
├── .env                       # Environment variables
├── requirements.txt           # Python dependencies
├── entrypoint.sh             # Container startup script
├── download_models.py        # Model download utility
├── Makefile                  # Helper commands
├── templates/
│   └── index.html            # Web interface
├── static/                   # Static assets
├── faiss_index_document_GDM/ # Vector database
└── temp/                     # Temporary files
```

## 🛠️ Commands

### Sử dụng Makefile (Recommended)
```bash
make build     # Build container
make up        # Start services
make down      # Stop services
make logs      # View logs
make shell     # Access container
make clean     # Clean up
make health    # Check status
```

### Sử dụng Docker Compose trực tiếp
```bash
# Development mode (với hot reload)
docker-compose up -d

# Production mode (không có override)
docker-compose -f docker-compose.yml up -d

# Xem logs
docker-compose logs -f

# Stop services
docker-compose down

# Clean up (cẩn thận - sẽ xóa volumes)
docker-compose down -v
```

## ⚙️ Configuration

### Environment Variables

Chỉnh sửa file `.env`:

```bash
# Ports
APP_PORT=80
OLLAMA_PORT=11434

# Models
OLLAMA_LLM_MODEL=qwen2.5:14b
OLLAMA_EMBEDDING_MODEL=bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# GPU (uncomment để sử dụng GPU cụ thể)
# CUDA_VISIBLE_DEVICES=0
```

### Custom Models

Để sử dụng models khác:

1. Cập nhật `.env` file
2. Rebuild container: `make build`
3. Start lại: `make up`

## 🐛 Troubleshooting

### Container không start
```bash
# Check logs
make logs

# Check status
make health

# Rebuild
make clean
make build
make up
```

### Models không load
```bash
# Access container
make shell

# Check Ollama models
ollama list

# Check HuggingFace cache
ls -la /chatbot_healthcare/models/
```

### Out of memory
- Tăng Docker memory limit
- Sử dụng smaller models
- Enable swap

### GPU không hoạt động
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# Check trong container
make shell
nvidia-smi
```

## 📊 Monitoring

### Health Checks
```bash
# Application health
curl http://localhost:80/ping

# Ollama health
curl http://localhost:11434/api/tags

# Container status
docker-compose ps
```

### Logs
```bash
# All services
make logs

# Specific service
docker-compose logs -f chatbot-healthcare
```

## 🔧 Development

### Local Development
```bash
# Start in dev mode (with file mounting)
make dev-up

# Modify files locally - changes will be reflected in container
# Hot reload enabled for Flask app
```

### Adding New Models
1. Update `download_models.py`
2. Modify `rag_inference.py`
3. Rebuild: `make build`

## 📝 API Endpoints

- `GET /` - Web interface
- `POST /ask` - Chat endpoint
- `POST /upload` - Audio upload
- `GET /ping` - Health check

## 🚀 FastAPI Features

### API Documentation
FastAPI tự động tạo API documentation tại:
- **Swagger UI**: `http://localhost:80/docs`
- **ReDoc**: `http://localhost:80/redoc`

### Development Mode
```bash
# Chạy trong development mode với auto-reload
uvicorn app:app --host 0.0.0.0 --port 80 --reload

# Hoặc chạy trực tiếp
python app.py
```

### Testing
```bash
# Test API endpoints
python test_app.py

# Hoặc sử dụng curl
curl -X GET http://localhost:80/ping
curl -X POST http://localhost:80/ask \
  -F "question=Hello" \
  -F "role=doctor" \
  -F "responseWithAudio=false"
```

### Performance Benefits
- **Async/Await**: Hỗ trợ bất đồng bộ native
- **High Performance**: Nhanh hơn Flask đáng kể
- **Type Safety**: Validation tự động với Pydantic
- **Auto Documentation**: OpenAPI/Swagger tự động

## 🤝 Contributing

1. Fork repository
2. Make changes
3. Test with `make dev-up`
4. Submit PR

## 📄 License

[Your license here]