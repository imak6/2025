from flask import Flask, render_template, request, jsonify, session
from chatbot import get_bot_response
import subprocess
import json
import requests
from flask_cors import CORS
import uuid

app = Flask(__name__)
CORS(app)
app.secret_key = 'mysupersecretkey'

@app.route("/")
def index():
    #return "Hello, Welcome to your AI Social Companion!"
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template("index.html", chat_history=session['chat_history'])

def query_local_model(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma:2b",
        "prompt": prompt,
        "stream": True   # Set stream to true to build response as it arrives, instead of waiting for few seconds/mins
    }
    # result = subprocess.run(
    #     ["ollama", "run", "mistral"],
    #     input=prompt.encode(),
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE
    # )

    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        response_text = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                response_text += data.get("response", "")
        # result = response.json()
        return response_text.strip()

    except Exception as e:
        return f"Error querying local model: {e}"

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", '')
    if not user_input:
        return jsonify({"response": "Please send a valid message."})
    # Add user input to chat history
    session['chat_history'].append({"role": "user", "content": user_input})
    session.modified = True # make sure Flask knows session has changed

    bot_response = query_local_model(user_input)

    # Add bot response to chat history
    session['chat_history'].append({"role": "assistant", "content": bot_response})
    session.modified = True
    print(f"Current chat history: {session['chat_history']}")
    return jsonify({"reply": bot_response})

@app.route('/clear')
def clear_chat():
    session.pop('chat_history', None)
    return 'Chat history cleared.', 200

if __name__ == "__main__":
    app.run(debug=True)

