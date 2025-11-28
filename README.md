Berikut adalah draft **README.md** lengkap yang sudah disesuaikan dengan konfigurasi `.env` Anda, struktur folder, dan link repository Torche.

Anda bisa langsung copy-paste kode di bawah ini ke dalam file `README.md` di project Anda.

-----

````markdown
# 🥗 Food Nutrition Virtual Assistant (Llama 3.1 Version)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi)
![Llama 3.1](https://img.shields.io/badge/AI-Llama%203.1-purple?style=for-the-badge)
![YOLOv8](https://img.shields.io/badge/Vision-YOLOv8-yellow?style=for-the-badge)

An intelligent, multimodal AI Agent designed to assist users with nutritional information in the Indonesian context. This project combines **Computer Vision (YOLO)** for food detection, **Vector Search (ChromaDB)** for factual grounding, and **Large Language Models (Llama 3.1)** for conversational advice.

## 🚀 Key Features

* **Multimodal Capabilities**: Analyzes both text queries and uploaded food images.
* **Local LLM Power**: Utilizes **Llama 3.1:8b** via Ollama for privacy and offline capability.
* **Context-Aware (RAG)**: Retrieves accurate nutrition data from local CSV datasets using **IndoBERT** embeddings.
* **Chat Memory**: Automatically manages conversation history using MongoDB.
* **Data Pipeline**: Includes ETL endpoints to sync new data into the Knowledge Base dynamically.

## 📋 Prerequisites

Before starting, ensure you have the following installed on your local machine:

1.  **Python 3.10+**
2.  **MongoDB Community Server** (Running on default port `27017`)
3.  **Ollama** (Running in the background)

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://git-devel.torche.id/food-detection-ai/food-nutrition-virtual-assistant-llama-3.1-ver.git](https://git-devel.torche.id/food-detection-ai/food-nutrition-virtual-assistant-llama-3.1-ver.git)
cd food-nutrition-virtual-assistant-llama-3.1-ver
````

### 2\. Environment Setup

Create and activate a virtual environment to isolate dependencies.

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Pull AI Models

Ensure Ollama is running, then pull the required LLM model:

```bash
ollama pull llama3.1:8b
```

### 5\. Configuration (.env)

Create a `.env` file in the root directory. Copy the configuration below exactly as it matches the project structure:

```ini
# --- Database Configuration ---
MONGO_CONNECTION_STRING=mongodb://localhost:27017/
HISTORY_DATABASE_STRING=chat_histories_db
HISTORY_COLLECTION_STRING=chats

# --- Vector Database (ChromaDB) ---
CHROMA_COLLECTION_NAME=food_knowledge
CHROMA_DB_PATH=src/ai_agent/components/database/chroma_db_store

# --- AI Models & Embeddings ---
LLM_MODEL_NAME=llama3.1:8b
# Using LazarusNLP IndoBERT for Indonesian context embedding
EMBEDDING_TEXT_MODEL_NAME=LazarusNLP/all-indobert-base-v4
YOLO_MODEL_PATH=models/food_detections/best.pt

# --- Data Sources & Prompts ---
KNOWLEDGE_BASE_JSON_SOURCE=data/raw/food_knowledge_indo.json
FOOD_DETAILS_PATH=assets/data/raw/all_food_details.csv
FOOD_NUTRITIONS_PATH=assets/data/raw/all_food_nutritions.csv
TEMPLATE_PROMPT_PATH=src/ai_agent/components/prompts/template_prompt.txt

# --- Image Processing Directories ---
RAW_DIR=assets/images/raw
ANNOTATED_DIR=assets/images/annotated
TEMP_DIR=assets/images/temp
CACHE_DIR=assets/images/temp/cache
```

> **⚠️ Important:** Ensure the YOLO model file exists at `models/food_detections/best.pt`.

## ▶️ Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

  * **API Documentation (Swagger UI):** [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)
  * **ReDoc:** [http://localhost:8000/redoc](https://www.google.com/search?q=http://localhost:8000/redoc)

## 📡 API Usage Guide

### 1\. Chat Interface (`POST /v1/chat`)

Interacts with the AI Agent.

  * **message**: User question (e.g., "Berapa kalori nasi goreng ini?").
  * **file**: (Optional) Image of the food.
  * **session\_id**: Leave empty to start a new session. The API will return a generated ID.

### 2\. Admin: Sync Knowledge (`POST /v1/admin/sync-knowledge`)

Triggers the background ETL process. It reads the CSV files specified in `.env`, generates embeddings using IndoBERT, and updates the ChromaDB vector store.

### 3\. Data Management (`POST /v1/upload/data`)

Allows appending new data to the raw CSV files (`details` or `nutritions`) to expand the knowledge base without accessing the server filesystem directly.

## 📂 Project Structure

```text
.
├── assets/
│   ├── data/raw/           # CSV Data Sources
│   └── images/             # Temp storage for image processing
├── models/
│   └── food_detections/    # YOLO .pt model
├── src/
│   ├── ai_agent/           # Logic: Orchestrator, Tools, Embeddings
│   ├── api/                # FastAPI Main Application
│   ├── config/             # Settings loader
│   └── knowledge/          # ETL Pipelines
├── .gitignore
├── requirements.txt
└── README.md
```

## 📝 License

Proprietary / Internal Use Only - **Torche Indonesia**

```
```
