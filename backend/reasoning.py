import ollama

def generate_reasoning(predicted_class, confidence):

    prompt = f"""
    MRI Brain Tumor Classification Result

    Predicted tumor type: {predicted_class}

    Confidence: {confidence*100:.2f}%

    Generate explanation for:
    1. What this tumor type means
    2. How it is spread, common characteristics and measures that help cure
    3. Why the AI model may have predicted this
    4. Mention that this is AI-assisted analysis and its trained on a less amount of data
    5. Recommend consultation with medical professionals

    Keep explanation simple and professional.
    """

    response = ollama.chat(
        model='llama3.2:1b',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    return response['message']['content']