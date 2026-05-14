from dotenv import load_dotenv
from groq import Groq
import os
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
def generate_reasoning(predicted_class, confidence):

    prompt = f"""
    MRI Brain Tumor Classification Result

    Predicted tumor type: {predicted_class}

    Confidence: {confidence*100:.2f}%

    Generate explanation for:
    
     What this tumor type means

     How it is spread and common characteristics

     Measures that helps cure

     Why the AI model may have predicted this

     Mention that this is AI-assisted analysis and its trained on a less amount of data

     Recommend consultation with medical professionals

    Keep explanation simple and professional.
    """

    response = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
        ,model='llama-3.1-8b-instant',
    )

    return response.choices[0].message.content