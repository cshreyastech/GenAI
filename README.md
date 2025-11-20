# 📘 Real Estate RAG System with LanceDB + OpenAI

An end-to-end Retrieval-Augmented Generation (RAG) pipeline that transforms real-estate listing data into searchable vector embeddings, retrieves relevant properties via semantic similarity, and generates grounded recommendations using an LLM.

This project demonstrates how to build a production-grade AI search & recommendation system combining:
LanceDB (vector database)
OpenAI Embeddings & Chat Models
Semantic Retrieval
RAG (Retrieval-Augmented Generation)
Deduplication, metadata normalization, and full-text transformation

## 🚀 Features
- Automatic JSON ingestion → metadata normalization → full-text generation
- Deterministic MD5 hashing to prevent duplicate entries
- OpenAI or SBERT embeddings (configurable)
- LanceDB vector storage + schema validation
- Optional HNSW vector index for fast search
- Semantic RAG querying with cosine similarity fallback
- LLM-generated recommendation with ranking & reasoning
- JSON-safe output suitable for API integration

## 📂 Project Structure
cd ./05-Building-GenAI-Solutions/04_project/ <br>
├── real_estate_db.py # LanceDB manager: ingestion, indexing, search <br>
├── embedding_utils.py # Embedding provider via OpenAI <br>
├── rag_pipeline.py # Retrieval + LLM answer generation <br>
├── run_pipeline.py # End-to-end runner <br>
├── listings.json # Sample real estate dataset <br>
└── README.md # Documentation <br>

## 🧠 Architecture Overview
High-Level Pipeline

              ┌────────────────────┐
              │  listings.json     │
              └─────────┬──────────┘
                        │ load
                        ▼
         ┌─────────────────────────────────┐
         │ RealEstateDBManager             │
         │ - normalize metadata            │
         │ - compose full_text             │
         │ - compute_md5_id()              │
         │ - deduplicate                   │
         └───────┬────────────────────────┘
                 │ embed
                 ▼
         ┌─────────────────────────────────┐
         │ Embedding Generator (OpenAI)    │
         └───────┬────────────────────────┘
                 │ store
                 ▼
        ┌──────────────────────────────────┐
        │ LanceDB (vector store)           │
        │ - schema validation              │
        │ - HNSW index (optional)          │
        └─────────┬────────────────────────┘
                  │ semantic search
                  ▼
           ┌─────────────────────┐
           │ RAGEngine           │
           │ Compose LLM prompt  │
           │ with retrieved docs │
           └─────────┬──────────┘
                     │
                     ▼
               ┌─────────────┐
               │   LLM        │
               │ Chat model   │
               └───────┬─────┘
                       │
                       ▼
            Final recommendation to user

## ⚙️ Installation
1. Clone the repository
git clone <your-repo-url>
cd real-estate-rag

2. Install dependencies
pip install -r requirements.txt

3. Add your OpenAI API key
data/GenAI/openai_key.txt

## 🏗️ How to Run the Pipeline
This script performs:
- Table creation (force or load existing)
- Ingestion + embedding
- Semantic search
- RAG ranking + LLM answer
python run_pipeline.py

Expected terminal output:
Created new LanceDB table: real_estate_listing
Added 10 new listings
Could not create index...
Result:
{ ... final RAG answer ... }

## 🔍 Query Example
user_query = "Looking for a modern 3-bedroom with good schools near public transit under $1.5M"
result = rag.query(user_query, k=5)
print(result["answer"])

Sample Generated Output
Top recommendations:
1. Oak Hill – $780,000 – excellent schools, modern layout
2. Brookstone – $950,000 – high rated schools, spacious
3. Sunnyvale – $450,000 – walkable & good transit access

Next steps:
- Verify school ratings
- Visit neighborhoods and compare commute access

## 🧰 Key Files Explained
real_estate_db.py
- Loads JSON
- Normalizes metadata
- Generates full_text summary per listing
- Computes deterministic IDs with MD5
- Stores vectors in LanceDB table using a Pydantic schema
- Deduplicates based on ID
- Supports native .search() or fallback cosine similarity

embedding_utils.py
- Loads OpenAI API key
- Provides embed_text_openai()
- Validates vector dimension
- Returns clean Python lists
- rag_pipeline.py
- Converts user query into an embedding
- Retrieves top-k matches
- Builds a reasoning prompt
- Calls OpenAI Chat Completions
- Returns structured final answer

run_pipeline.py
Main orchestrator:
- creates table
- ingests listings
- builds vector index
- runs sample query

## 📊 Results

The system successfully:
- Ingests 10 real-estate listings
- Creates stable embeddings
- Retrieves contextually relevant listings
- Generates realistic, grounded recommendations
- Explains reasoning and provides actionable next steps

This pipeline can be extended into:
- A real estate chatbot
- A search API
- A web frontend using Streamlit or Next.js
- A production RAG backend using FastAPI