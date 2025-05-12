# 🧠 Agent-Based Automated Data Analysis System

This project is a modular, intelligent data analysis system powered by **LangChain agents** and **local open-source LLMs** (like Mistral via Ollama). It can ingest a wide range of datasets, profile them, visualize trends, and generate human-readable insights automatically.

## Project Structure

DATA_ANALYSIS_AGENT/
├── app.py
├── main.py
├── requirements.txt
├── data/...
└── docker/
    └── app.dockerfile

---

## 📌 Features

- Load any tabular dataset (CSV, Excel, JSON, Parquet)
- Automatically profile and clean data
- Visualize key features and distributions
- Generate text-based summaries using LLMs
- Built with modular agents (LangChain)
- Works offline using local models (Mistral, LLaMA, etc.)

---

## 🧱 Architecture

```text
┌──────────────────────────┐
│  User Input / Dataset    │
└────────────┬─────────────┘
             │
      (Dataset Loader Agent)
             ▼
┌──────────────────────────┐
│ Data Ingestion & Format  │
│ (CSV, Excel, Parquet...) │
└────────────┬─────────────┘
             │
   (Data Profiler Agent)
             ▼
┌────────────────────────────┐
│ Profiling (types, nulls,   │
│ basic stats, cardinality)  │
└────────────┬───────────────┘
             │
     (Cleaning Agent)
             ▼
┌────────────────────────────┐
│ Suggestions/fixes (nulls,  │
│ outliers, type mismatches) │
└────────────┬───────────────┘
             │
   (EDA / Viz Agent)
             ▼
┌────────────────────────────┐
│ Generate plots, correlations│
│ distributions, time trends │
└────────────┬───────────────┘
             │
    (Insight Agent - LLM)
             ▼
┌────────────────────────────┐
│ Summary, anomalies, trends │
│ (LLM-generated text)       │
└────────────┬───────────────┘
             │
      Optional: Modeling Agent
             ▼
┌────────────────────────────┐
│ ML modeling, AutoML, etc.  │
└────────────────────────────┘
```

## Requirements

Install using pip:

```
pip install langchain-community pandas openai pandas-profiling seaborn matplotlib streamlit
```

Local LLM (via Ollama)

macos: brew install ollama
others: <https://ollama.com>

Download and run a local model (e.g: Mistral, gemma:2b, qwen2.5)
```
"brew services start ollama"
"ollama run gemma:2b"
```
## Steps that get executed

- Define the LangChain agent pipeline using a local LLM via Ollama.

This script will:

- Load a dataset

- Run basic profiling

- Pass a data summary to the LLM agent (via LangChain)

- Output insights or suggestions

## Add Visualizations agent

- This agent will generate and optionally save visualizations (histograms, correlation heatmaps, bar plots)

## DOCKER INSTRUCTIONS TO HOST THE STREAMLIT APP

step1: Build docker image using the dockerfile
```
docker build -f docker/app.dockerfile -t data-anaysis-agent .
```  

step2: Run the Container (streamlit app)
```
docker run -p 8501:8501 data-agent-app
```
```
visit <http://localhost:8501>
```
step3: Push the created docker image to a repository (dockerhub/github container registry)
```
docker tag data-analysis-agent username/data-analysis-agent:latest
```

step4: login to docker hub
```
docker login
```

step5: Push the image
```
docker push username/data-analysis-agent:latest
```

step6: Run from the repository (Optional)
From any machine with docker installed:
**docker pull username/data-analysis-agent:latest** - pull the image pushed to docker hub

**docker run -p 8501:8501 username/data-analysis-agent:latest** - run the app/container
