from transformers import pipeline

def main():
    messages = [
        {"role": "user", "content": "Who are you? Why should I use AI?"}
    ]
    
    # Extract the user's input as a string.
    user_input = messages[0]["content"]

    # Create the text-generation pipeline using a model such as distilgpt2.
    # Adjust 'device' as needed (use 0 for GPU or -1 for CPU).
    text_generator = pipeline(
        "text-generation", 
        model="distilgpt2", 
        device=0  # Change to -1 if you do not have a GPU.
    )
    
    # Generate a response with a maximum of 100 tokens.
    response = text_generator(user_input, max_length=100)
    
    print("Response:")
    print(response)

if __name__ == "__main__":
    main()
