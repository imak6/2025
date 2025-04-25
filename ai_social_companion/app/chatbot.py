
def get_bot_response(message):
    # Temporary rule-based logic
    message = message.lower()
    if "hello" in message or "hi" in message:
        return "Hi there! How can I help you today?"
    elif "help" in message:
        return "Sure! I'm here to assist you. What do you need help with?"
    else:
        return "I'm still learning. Could you please rephrase that?"

