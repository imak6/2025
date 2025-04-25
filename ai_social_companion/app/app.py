from flask import Flask, render_template, request, jsonify
from chatbot import get_bot_response
import subprocess
import json
import requests

app = Flask(__name__)

@app.route("/")
def home():
    #return "Hello, Welcome to your AI Social Companion!"
    return render_template("index.html")

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
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"response": "Please send a valid message."})

    bot_response = query_local_model(user_input)
    return jsonify({"reply": bot_response})


if __name__ == "__main__":
    app.run(debug=True)

