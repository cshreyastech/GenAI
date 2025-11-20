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
.
├── real_estate_db.py # LanceDB manager: ingestion, indexing, search
├── embedding_utils.py # Embedding provider via OpenAI
├── rag_pipeline.py # Retrieval + LLM answer generation
├── run_pipeline.py # End-to-end runner
├── listings.json # Sample real estate dataset
└── README.md # Documentation
