import openai
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)

# Define the conversation with roles and content
conversation = [
    {"role": "system", "content": "You are a knowledgeable Python programming assistant who explains concepts clearly."},
    {"role": "user", "content": "Can you explain how virtual environments (venv) work in Python?"},
    {"role": "assistant", "content": "Sure! A virtual environment (venv) in Python is an isolated workspace where you can install packages specific to a project without affecting the global Python environment."},
    {"role": "user", "content": "Why is using a virtual environment important?"}
]

# Make a single API call with all roles using the updated method
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=conversation
)

# Display the conversation with roles and content
print("\n--- Conversation History ---")
for message in conversation:
    print(f"\nRole: {message['role'].capitalize()}")
    print(f"Content: {message['content']}")

# Display the AI's latest response
print("\n--- Assistant's Response ---")
print(response.choices[0].message.content)


# Output:
# --- Conversation History ---

# Role: System
# Content: You are a knowledgeable Python programming assistant who explains concepts clearly.

# Role: User
# Content: Can you explain how virtual environments (venv) work in Python?

# Role: Assistant
# Content: Sure! A virtual environment (venv) in Python is an isolated workspace where you can install packages specific to a project without affecting the global Python environment.

# Role: User
# Content: Why is using a virtual environment important?

# --- Assistant's Response ---
# Using a virtual environment is important because it allows you to manage project dependencies and avoid conflicts between different projects that may require different versions of the same packages. It also helps to keep your global Python environment clean and avoids potential issues when working on multiple projects.