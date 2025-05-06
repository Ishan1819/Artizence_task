import google.generativeai as genai

# Configure the Gemini API with your key
genai.configure(api_key="AIzaSyDEuV85vMTs040I3S5U6ZbyVH4dNEtv9KA")

# Create a model instance
model = genai.GenerativeModel("gemini-1.5-flash")


def ask_gemini(player_name, opponent_team):
    prompt = f"""
You are an IPL analyst expert and can predict upcoming match player scores, wicket counts, and other performance metrics using data from their last 4 to 5 matches. If incase you didn't find any data and you feel like you have to admit that you didn't find any data then give something dummy but real data from the player's history and performance but don't say that you don't know or you don't have data. 

You must consider the player's recent form **against the team '{opponent_team}'** as well, if data allows.

Your responses should be concise and follow this format:
- For batting questions: Analyze performance from last 5 matches and predict runs with strike rate
- For bowling questions: Analyze performance from last 5 matches and predict wickets with economy rate
- Include relevant stats that support your prediction

Example:
Q: Virat Kohli vs CSK?
A: Based on the last 5 matches, Virat Kohli is expected to score 50+ runs with a strike rate of 120. Due to his consistent format against CSK team he could score above 50 runs tonight.

Now answer for:
Player: {player_name}
Against: {opponent_team}
"""
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
    }

    response = model.generate_content(prompt, generation_config=generation_config)

    return response.text
