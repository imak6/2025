---
id: ai-social-companion
title: AI Social Companion
sidebar_position: 1
---

# 🧠 AI Social Companion Flask Prototype

Welcome to the documentation for the **AI Social Companion**, an open-source Flask web app powered by a local LLM. This app is designed as a foundational prototype for building interactive, personalized assistants that can serve as companions—especially for users with accessibility needs.

---

## 🚀 Overview

This prototype demonstrates:

- A basic chatbot interface served via Flask
- Integration with **local** LLMs like `gemma:2b` using **Ollama**
- A simple user interface to exchange messages with the assistant

---

## 🔧 Features

### ✅ Flask Web Server

- Endpoint: `/chat` accepts POST messages and returns the AI's reply

### ✅ LLM Backend Integration

- Calls local LLM through `http://localhost:11434/api/generate`
- Supports **streamed responses** for faster interactions
- Uses `gemma:2b` (configurable)

### ✅ Simple HTML Chat UI

- Input and display for user and bot messages
- Uses `fetch` API to communicate with the backend in real time

---

## 🗂 File Structure

ai-companion
├── app
│   ├── static
│   │   └── style.css
│   ├── templates
│   │   └── index.html
│   ├── chatbot.py
│   └── app.py
├── README.md
└── requirements.txt

---

## 📦 Setup & Usage

### 🧩 1. Install Ollama and Pull Model

ollama pull gemma:2b

### Run the model locally

ollama run gemma:2b

### Start Flask Server

python app.py

### Access in Browser

Open: [flask_app]

[flask_app]: http://localhost:5000/
