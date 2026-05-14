# Brain Tumor Detection and AI Reasoning System

## Overview

This project is an AI-powered brain tumor detection system that combines:

- Deep Learning based MRI image classification
- FastAPI backend for inference APIs
- Streamlit frontend for user interaction
- LLM-powered medical reasoning using Groq API
- Docker and Docker Compose based deployment
- AWS EC2 deployment support

The system allows users to upload MRI brain scan images and receive:
- Predicted tumor classification
- Confidence score
- AI-generated reasoning and explanation

---

# Features

- MRI Brain Tumor Detection using TensorFlow/Keras
- FastAPI REST API backend
- Streamlit interactive frontend
- Groq LLM integration for reasoning
- Dockerized architecture
- Docker Compose multi-container setup
- Environment variable support using `.env`
- AWS EC2 deployment ready

---

# Tech Stack

## Backend
- Python
- FastAPI
- TensorFlow
- NumPy
- OpenCV
- Groq API

## Frontend
- Streamlit
- Requests
- Pillow

## Deployment
- Docker
- Docker Compose
- AWS EC2

---

# Project Structure

```bash
project/
│
├── docker-compose.yml
├── .gitignore
├── README.md
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── predictor.py
│   ├── reasoning.py
│   ├── Brain_tumour_model.h5
│   └── .env
│
└── frontend/
    ├── Dockerfile
    ├── requirements.txt
    └── frontend.py
```

---

# Backend Requirements

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
numpy==1.26.4
opencv-python-headless==4.10.0.84
tensorflow==2.17.0
python-multipart==0.0.9
python-dotenv==1.0.1
groq==0.11.0
```

---

# Frontend Requirements

```txt
streamlit==1.39.0
requests==2.32.3
pillow==10.4.0
```

---

# Environment Variables

Create a `.env` file inside the `backend/` directory:

```env
GROQ_API_KEY=your_api_key_here
```

---

# Docker Setup

## Build and Run Containers

From the project root directory:

```bash
docker compose up --build
```

---

# Access the Application

## Frontend
```txt
http://localhost:8501
```

## Backend API
```txt
http://localhost:8000
```

## FastAPI Swagger Docs
```txt
http://localhost:8000/docs
```

---

# Backend Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

# Frontend Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

---

# Docker Compose Configuration

```yaml
version: '3.9'

services:

  backend:
    build: ./backend
    container_name: brain-backend

    ports:
      - "8000:8000"

    env_file:
      - ./backend/.env

    restart: always

  frontend:
    build: ./frontend
    container_name: brain-frontend

    ports:
      - "8501:8501"

    depends_on:
      - backend

    restart: always
```

---

# AWS EC2 Deployment

## Steps

1. Launch Ubuntu EC2 instance
2. Install Docker and Docker Compose
3. Clone the repository
4. Configure `.env`
5. Run:

```bash
docker compose up --build -d
```

6. Open EC2 Security Group ports:
- 22 (SSH)
- 8000 (FastAPI)
- 8501 (Streamlit)

---

# API Endpoint

## Prediction Endpoint

```http
POST /predict
```

Upload MRI image and receive prediction result.

---

# Future Improvements

- TensorFlow Lite optimization
- GPU inference support
- Authentication system
- Database integration
- Model monitoring
- CI/CD pipeline
- Kubernetes deployment

---

# Disclaimer

This project is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis.

---

# Author

Aswin Santhosh

BTech Computer Science Engineering  
AI/ML Development Enthusiast
