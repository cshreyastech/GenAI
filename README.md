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
High-Level Pipeline <br>
              ┌────────────────────┐ <br>
              │  listings.json     │ <br>
              └─────────┬──────────┘ <br>
                        │ load  <br>
                        ▼ <br>
         ┌─────────────────────────────────┐ <br>
         │ RealEstateDBManager             │ <br>
         │ - normalize metadata            │ <br>
         │ - compose full_text             │ <br>
         │ - compute_md5_id()              │ <br>
         │ - deduplicate                   │ <br>
         └───────┬────────────────────────┘ <br>
                 │ embed <br>
                 ▼ <br>
         ┌─────────────────────────────────┐ <br>
         │ Embedding Generator (OpenAI)    │ <br>
         └───────┬────────────────────────┘ <br>
                 │ store <br>
                 ▼ <br>
        ┌──────────────────────────────────┐ <br>
        │ LanceDB (vector store)           │ <br>
        │ - schema validation              │ <br>
        │ - HNSW index (optional)          │ <br>
        └─────────┬────────────────────────┘ <br>
                  │ semantic search <br>
                  ▼ <br>
           ┌─────────────────────┐ <br>
           │ RAGEngine           │ <br>
           │ Compose LLM prompt  │ <br>
           │ with retrieved docs │ <br>
           └─────────┬──────────┘ <br>
                     │ <br>
                     ▼ <br>
               ┌─────────────┐ <br>
               │   LLM        │ <br>
               │ Chat model   │ <br>
               └───────┬─────┘ <br>
                       │ <br>
                       ▼ <br>
            Final recommendation to user <br>
