# 🧠 Goalie - Your Friendly Career Buddy ⚽

**Goalie** is a fun and friendly AI-powered career guide that chats with you, understands your interests, and recommends a career path you might love! It’s like having a chill mentor who speaks your language (and uses emojis sometimes 👀).

---

## 🚀 Features

- 🧠 Uses AI embeddings (via Mistral) to understand your personality and interests
- 🧭 Matches you to a career path (STEM, Arts, Sports, etc.)
- 🎙️ Responds in a supportive and funny tone (think: your hype buddy!)
- 💬 Remembers chat history (optional: with LangChain integration)
- 🌐 Flask-powered web interface for conversations

---

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **AI Embeddings & Chat**: Mistral (via `mistralai`)
- **Similarity Matching**: Cosine Similarity (`scikit-learn`)
- **Memory (Optional)**: LangChain
- **Frontend**: HTML/CSS (custom styling)

---

## 🧪 How It Works

1. User types something like:
   > "I love building robots and doing math puzzles"
2. The input is converted into an embedding using `mistral-embed`
3. It’s compared (using cosine similarity) with predefined descriptions of career categories
4. The best match is selected (e.g., **STEM**)
5. The chatbot (Goalie 🧢) explains the result with a personalized, encouraging message

---

## 🖥️ Local Setup

```bash
# Clone the repo
git clone https://github.com/Vidushi2709/Goalie.git
cd goalie

# Set up virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Add your Mistral API key to .env
echo MISTRAL_API=your_key_here > .env

# Run the app
python app.py
```
---

Created by Vin ❤️
