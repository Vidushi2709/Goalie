from flask import Flask, request, jsonify, render_template
from chatbot import best_career, explain, clarify
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import os 

load_dotenv()

app = Flask(__name__, static_folder="../static", template_folder="../templates")

memory = ConversationBufferMemory(return_messages=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/goalie", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")

    try:
        memory.chat_memory.add_user_message(HumanMessage(content=user_input))
        if user_input.lower() in ["exit", "quit"]:
            return jsonify({"reply": "🤖 Goalie: That’s a wrap, superstar! Go chase those dreams. 🏁 \n"})

        career = best_career(user_input)
        if career:
            response = explain(user_input, career, memory)
        else:
            response = clarify()
        memory.chat_memory.add_ai_message(AIMessage(content=response))

        return jsonify({"reply": response})

    except Exception as e:
        return jsonify({"reply": f"⚠️ Error: {str(e)}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
