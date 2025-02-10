from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

response = client.images.generate(
    model="dall-e-3",
    prompt="""
    Create a flyer for the Stony Brook Computing Society's upcoming event.
    Let there be a space for the time and date.

    Have some rubber ducks in the middle of the flyer be cartoonish and colorful 
    with glowing neon circuits and a sleek, robotic design.

    Use a similar color scheme to blue and yellow, maintaining a clean, minimalistic layout.
    """,
    n=1,
    size="1024x1024",
    response_format="url"
)

print(response.data[0].url)
