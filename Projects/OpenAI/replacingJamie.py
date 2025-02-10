from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


styles = ["cyberpunk", "retro 80s", "minimalistic", "comic book style", "terminator-style"]

for style in styles:
    response = client.images.generate(
        model="dall-e-3",
        prompt=f"""
        Create a flyer for the Stony Brook Computing Society's upcoming workshop.
        The flyer should feature cartoonish rubber ducks with neon circuits, in a {style} aesthetic.
        Leave space for event details like the time and date.
        """,
        n=1,
        size="1024x1024",
        response_format="url"
    )
    print(f"{style} style flyer: {response.data[0].url}")
