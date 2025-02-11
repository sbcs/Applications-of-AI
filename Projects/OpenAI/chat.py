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
print("DISPLAYING RESPONSE with different ROLES")
# Display the conversation with roles and content
print("\n--- Conversation History ---")
for message in conversation:
    print(f"\nRole: {message['role'].capitalize()}")
    print(f"Content: {message['content']}")

# Display the AI's latest response
print("\n--- Assistant's Response ---")
print(response.choices[0].message.content)

# Function to make API call with variable temperature
def get_response_with_temperature(temp):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        temperature=temp
    )
    return response.choices[0].message.content

print("DISPLAYING RESPONSES WITH DIFFERENT TEMPERATURES")
# Display the conversation with roles and content
print("\n--- Conversation History ---")
for message in conversation:
    print(f"\nRole: {message['role'].capitalize()}")
    print(f"Content: {message['content']}")

# Test with different temperature settings
temperatures = [0.2, 0.7, 1.2, 1.9]
for temp in temperatures:
    print(f"\n--- Assistant's Response with Temperature {temp} ---")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Write a short story about how a monkey ended up colonizing the moon"}
        ],
        max_tokens=400,
        temperature=temp
    )
    response_content = response.choices[0].message.content
    print(f"For the temperature of {temp}, the response is:\n{response_content}")
    print("--------------------")

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
