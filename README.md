# 🌿 Darukaa.Earth: AI Biodiversity Intelligence

🔍 *Generative AI • RAG • FastAPI • React • Vector Database • Environmental Intelligence*

## 🚀 Tech Stack & Domains

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)
![React](https://img.shields.io/badge/Frontend-React-61DAFB?logo=react)
![TypeScript](https://img.shields.io/badge/Language-TypeScript-3178C6?logo=typescript)
![Gemini](https://img.shields.io/badge/LLM-Gemini-4285F4?logo=google)
![RAG](https://img.shields.io/badge/AI-RAG-purple)
![Vector%20Database](https://img.shields.io/badge/Database-Vector%20Database-orange)
![Vercel](https://img.shields.io/badge/Deployment-Vercel-black?logo=vercel)
![Domain](https://img.shields.io/badge/Domain-Biodiversity%20%26%20Environmental%20AI-green)

---

## 📘 Overview

This project is an **AI-powered Biodiversity Intelligence system** designed to analyze environmental conditions and generate **actionable, evidence-backed recommendations** for improving biodiversity.

The system combines **structured environmental data, natural-language conversations, scientific knowledge retrieval, and multi-metric reasoning** to understand relationships between soil health, land use, biodiversity, climate, and human impact.

Unlike a generic LLM chatbot, the system uses a **retrievable scientific knowledge layer** to ground recommendations in research papers, reports, and environmental datasets.

The system is designed to behave more like an **AI environmental scientist than a chatbot**.

---

## 🎯 Problem Statement

Environmental problems are usually influenced by multiple interconnected factors.

For example:

- Low rainfall can affect water availability
- Low soil organic carbon can affect soil health
- Monoculture can reduce habitat diversity
- Habitat changes can affect biodiversity

A generic chatbot may provide a recommendation such as:

> "Use sustainable agricultural practices."

This project goes further by connecting multiple environmental variables and providing:

- What to do
- Why it works
- Which metrics may improve
- Expected time horizon
- Scientific evidence supporting the recommendation

---

## 🌍 Environmental Metrics

| Category | Metrics |
|----------|---------|
| 🌱 Soil Health | pH, Organic Carbon, Moisture |
| 🌾 Land Use / Land Cover | Crop Type, Land Use |
| 🐝 Biodiversity | Species Richness, Habitat Diversity |
| 🌦️ Climate | Temperature, Rainfall |
| 🏭 Human Impact | Pollution, Deforestation |

---

## 🧠 Core Features

### 🔎 Scientific Knowledge Retrieval

The system maintains a retrievable knowledge layer containing relevant:

- Research papers
- Scientific reports
- Environmental datasets

The knowledge pipeline follows:

```text
Scientific Documents
        ↓
Document Processing
        ↓
Chunking
        ↓
Embeddings
        ↓
Vector Database
        ↓
Semantic Retrieval
        ↓
Relevant Evidence
        ↓
AI Reasoning
