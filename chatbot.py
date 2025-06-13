import os
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatCompletionResponseChoice
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.conversation.memory import ConversationBufferMemory

message_history = ChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=message_history,
    return_messages=True
)


load_dotenv()
mistral_api_key= os.getenv("MISTRAL_API")

client= MistralClient(api_key=mistral_api_key)

chat_model= 'mistral-small-latest'
embedding_model= 'mistral-embed'

labels = {
    "STEM": "solving problems, coding, robotics, math, building things, science experiments, AI, engineering, hacking, space exploration, 3D printing, physics, puzzles, gaming logic, apps & tech gadgets",
    "Arts & Media": "drawing, painting, music, acting, storytelling, creating content, animation, graphic design, photography, filmmaking, fashion, writing blogs or poetry, editing reels, theatre, performing arts, podcasts",
    "Business": "entrepreneurship, finance, starting a business, leadership, strategy, marketing, sales, management, branding, pitch decks, investing, stock market, running campaigns, building teams",
    "Health & Medicine": "helping people, biology, health, becoming a doctor or nurse, mental health awareness, nutrition, healthcare innovation, anatomy, physical therapy, first aid, medical research, fitness science",
    "Sports": "playing sports, fitness, training, coaching, competition, team spirit, outdoor games, swimming, athletics, cricket, football, gym workouts, adventure sports, tournaments, strategy games",
    "Environment": "nature, animals, climate change, protecting forests, recycling, sustainability, planting trees, saving water, wildlife conservation, eco-tech, marine life, green energy, environmental science",
    "Social Impact & Law": "public speaking, debating, politics, law, human rights, volunteering, activism, social justice, Model UN, helping communities, legal drama shows, changing the world",
    "Education & Psychology": "teaching others, mentoring, tutoring, understanding how people think, child psychology, learning styles, academic motivation, counseling, special education, mind & behavior",
    "Technology & Gaming": "video games, e-sports, game design, building gaming PCs, VR, AR, tech reviews, streaming games, game logic, level design, competitive gaming, game psychology"
}

def get_embeddings(x):
    response= client.embeddings(input=[x], model= embedding_model)
    return response.data[0].embedding 

def best_career(user_ip):
    ip_vec= get_embeddings(user_ip)
    best_match= None
    best_score= -1

    for career, desc in labels.items():
        desc_vec= get_embeddings(desc)
        sim= cosine_similarity([ip_vec], [desc_vec])[0][0]
        if sim > best_score:
            best_score= sim
            best_match= career
    
    return best_match

def explain(user_ip, career, memory):
    system_message = ChatMessage(
        role="system",
        content='''You are Goalie, a caring, funny, and relaxed elder sibling who gives career advice like a heartfelt pep talk. You don’t lecture or overwhelm. Instead, you listen, reflect, and guide gently. You’re the kind of sibling who stays up late helping someone figure out their dreams, cracks a few jokes to lighten the mood, and reminds them it’s okay to be unsure.
    You speak casually, like a real person, never robotic or overly formal. You offer clarity, encouragement, and relatable advice, even when the user is confused, anxious, or lost. Your tone is warm, real, and honest always aiming to reduce pressure and increase confidence.
    Above all, you root for growth over perfection. Help the user explore their interests, hype them up when they feel unsure, and remind them they’re not alone in figuring things out. '''
    )

    # Convert LangChain memory messages to Mistral-compatible format
    history_msgs = []
    for m in memory.chat_memory.messages:
        if isinstance(m, HumanMessage):
            history_msgs.append(ChatMessage(role="user", content=m.content))
        elif isinstance(m, AIMessage):
            history_msgs.append(ChatMessage(role="assistant", content=m.content))

    # Construct your tailored prompt based on latest input
    final_prompt = f"""
    🎤 Yo! It's your career buddy, **Goalie** — here to drop some wisdom without the boring vibes.

    So you told me: "{user_ip}", right?

    Well guess what? Based on that, I think you’d totally vibe with **{career.upper()}**. 🎯

    Why? Because this path is all about: {labels[career]} — and that sounds *exactly* like your jam.

    Here’s why it’s pretty cool:
    - You get to do stuff you actually enjoy.
    - You’ll probably meet other weirdly awesome people like you.
    - There’s no fixed playbook — just follow what clicks, experiment a little, and you’re golden.

    🎮 Think of life like a game. You don’t have to speedrun it. Explore the map, try side quests, and don’t stress if you haven’t “unlocked” your final job title yet.

    I’m here for the long haul, so whenever you need to figure things out — just pass me the ball. ⚽️😎
    """

    # Add the final user prompt to message history
    history_msgs.append(ChatMessage(role="user", content=final_prompt))

    # Include the system message + chat history
    messages = [system_message] + history_msgs

    # Call Mistral with full context
    response = client.chat(model=chat_model, messages=messages)
    return response.choices[0].message.content.strip()


def clarify():
    return "What gets your brain buzzing or your heart racing? Tell me your hobbies, fav subjects, or anything you love!"

'''
def Goal():
    print("Hi there! I’m Goalie - your personal career keeper. Kick me your interests, and I’ll block out the confusion!")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("🤖 Goalie: That’s a wrap, superstar! Go chase those dreams. 🏁")
            break
        memory.chat_memory.add_user_message(HumanMessage(content=user_input))

        try:
            career = best_career(user_input)
            if career:
                response = explain(user_input, career)
            else:
                response = clarify()
            memory.chat_memory.add_ai_message(AIMessage(content=response))
            print(f"\n 🤖 Goalie: {response}\n")

        except Exception as e:
            print("⚠️ Error:", e)

Goal()
'''