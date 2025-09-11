
import json
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.deepseek.com")

prompt = f"""
You are an image description assistant. Assume you are currently watching a video, and I will give you a question related to the video. 
Your task is to generate potential image caption text based on the question.

Core requirements:
1. The output must be concise, objective, and visually observable facts.
2. Exclude subjective judgments, invisible information, and the specific content the question is asking.
3. Avoid using quantities; use implicit references instead.
4. Keep the output within 30 words.

Output format:
Directly output the visual description without any explanations or annotations.


Here is the question:
"""

def chatcompletions(question = ""):
    """LLM
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt + question + "\nOutput Caption:"},
        ],
        stream=False
    )

    out = json.loads(response)['choices'][0]['message']['content']

    return out

