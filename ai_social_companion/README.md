# AI Social Companion

A Flask-based conversational AI prototype designed to reduce isolation among people with disabilities.

## Features to implement

- Text and voice input
- Emotion-aware responses
- Customizable chatbot personality

## Getting Started

1. Clone the repo
2. Run `python app.py`
3. Visit `http://localhost:5000`

## AI Social Companion – Flask + Local Model (Mistral/gemma:2b/tinyllama via Ollama)

## 🔧 Setup

(Backend: Flask handles chat logic.

Local LLM (Mistral/tinyllama/gemma:2b) runs via Ollama subprocess.

Simple UI sends user input to backend via /chat endpoint.)

## 📦 Features Implemented (So Far)

### ✅ 1. Flask Web App

- Basic Flask server to host the chatbot.
- `/` endpoint renders an HTML interface (`index.html`).
- `/chat` endpoint handles POST requests with user messages and returns the model's response.

### ✅ 2. Local LLM Integration (via Ollama)

- Uses `requests.post` to call Ollama’s HTTP API at `http://localhost:11434/api/generate`.
- Model: currently configured for `gemma:2b` (tried mistral and tinyllama previously)(streaming mode supported).
- Handles response in streaming chunks for smoother interactions.

### ✅ 3. HTML Chat UI (Basic)

- HTML file `templates/index.html` allows user input and shows bot responses.
- JavaScript sends chat messages to the Flask `/chat` endpoint using fetch.

---

## 🗂️ File Structure

ai-companion/
├── app/
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   └── index.html
│   ├── chatbot.py
│   └── app.py
├── README.md
└── requirements.txt

---

## ⚙️ How to Run Locally

### 1. Install Ollama and Pull the Model

```bash```
ollama pull gemma:2b

### 2. Start Ollama Runtime

ollama run gemma:2b

### 3. Start the Flask App

python app.py

### 4. Open in Browser

Visit <http://127.0.0.1:5000/>

## Feature: Chat History (per session)

### Step1

Use Flask's "session" to store chat history temporarily.

1. Enable session management.
2. Store user/AI messages as a list in the session.

3. Display Chat history in the UI (update index.html and add styling in .css)

### Persist chat history beyond session memory

Use SQLite ot JSON files and then  implement actual LLM memory (context window/summaries).

### Next Steps (Planned Features)

1. 🎙️ Voice Support (input/output - TTS+STT)

2. 🎭 Personality customization

3. 🌙 Chat UI enhancements (dark mode,
   editable messages, transitions)

4. 🧠 Memory & history per user

5. Language support toggle
