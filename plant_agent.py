# plant_agent.py

import os
from PIL import Image
from LLM import prompt_gemini  # Import your wrapper

# JSON schema for structured output from the LLM
SCHEMA = {
    "type": "object",
    "properties": {
        "latin_name": {"type": "string"},
        "common_name": {"type": "string"},
        "water_ml": {"type": "number"},
        "confidence": {"type": "number"}
    },
    "required": ["latin_name", "water_ml"]
}

def analyze_plant(image_path: str):
    image_path = image_path = "C:/Users/Odin/Documents/GitHub/robotics_final_project/images/monstera-deliciosa.jpg"

    img = Image.open(image_path)
    
    prompt_text = (
        "You are an expert botanist. Identify the plant in the image. "
        "Return the plant's scientific (Latin) name, common name, "
        "recommended water amount in milliliters, and a confidence "
        "score from 0 to 1. Respond strictly following the JSON schema."
    )

    # IMPORTANT: same usage style as your old prompt_gemini calls
    response, logs = prompt_gemini(
        input_prompt=[img, prompt_text],
        schema=SCHEMA,
        with_tokens_info=True,
        temperature=0.0
    )

    return response, logs


if __name__ == "__main__":
    # Test the system with the example image
    test_image = "images/monstera-deliciosa.jpg"

    result = analyze_plant(test_image)

    if result:
        print("\nPlant Analysis Result:")
        print(result)
    else:
        print("No response received from the LLM.")

