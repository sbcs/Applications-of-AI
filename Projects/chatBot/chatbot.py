from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv  # Import the dotenv module to load env variables
from openai import OpenAI
from transformers import pipeline

# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No API key provided. Please set the OPENAI_API_KEY environment variable in your .env file.")

# Initialize the OpenAI client with the API key from the environment
client = OpenAI(api_key=api_key)
sentiment_analyzer = pipeline("sentiment-analysis")
app = Flask(__name__)

EDUCATION_PROMPT = """You are an AI teaching assistant specializing in explaining:
- AI fundamentals and applications
- Hugging Face's open-source models
- OpenAI's commercial APIs
- Practical AI implementations

Provide clear, concise and short example-driven explanations. For comparisons:
- Hugging Face: Open-source, self-hosted models
- OpenAI: Commercial API services
- Applications: Use cases in healthcare, finance, education, etc."""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_input = data.get("user_input", "").strip()
    mode = data.get("mode", "")
    topic = data.get("topic", "")
    demo_option = data.get("demo_option", "")

    try:
        if not user_input:
            return jsonify({"error": "Please enter a question"}), 400

        if topic == "AI Overview":
            if len(user_input.split()) < 2:
                system_message = EDUCATION_PROMPT + " Focus on generic real-world applications."
            else:
                system_message = f"{EDUCATION_PROMPT} Focus on real-world applications in {user_input} industry."

            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": system_message}],
                max_tokens=300
            )

            result = response.choices[0].message.content

        elif topic == "Hugging Face Applications":
            analysis = sentiment_analyzer(user_input)[0]
            result = (
                f"Sentiment Analysis Result:\n"
                f"Text: {user_input}\n"
                f"Label: {analysis['label']}\n"
                f"Confidence: {analysis['score']:.2f}"
            )

        elif topic == "OpenAI Applications":
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{
                    "role": "system",
                    "content": f"Explain OpenAI's role in {user_input} with technical details"
                }],
                max_tokens=300
            )
            result = response.choices[0].message.content

        elif topic == "Hands-on Demo":
            if demo_option == "Generate Text":
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": user_input}],
                    max_tokens=200
                )
                result = response.choices[0].message.content
            
            elif demo_option == "Analyze Sentiment":
                analysis = sentiment_analyzer(user_input)[0]
                result = f"Sentiment: {analysis['label']} (Score: {analysis['score']:.2f})"
            
            elif demo_option == "AI Image Generation":
                result = "Image generation requires DALL-E 3 API setup"

        else:
            result = "Please select a valid topic"

        return jsonify({"response": result})

    except Exception as e:
        return jsonify({"error": f"AI service error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=False)
