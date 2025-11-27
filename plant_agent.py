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
    """
    Sends the plant image to the LLM and retrieves structured plant info.
    """

    # Load image
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Prompt instructions for Gemini
    prompt_text = (
        "You are an expert botanist. Identify the plant in the image. "
        "Return the plant's scientific (Latin) name, its common name, "
        "and estimate how much water (in milliliters) a typical healthy plant "
        "of this type should receive per watering. "
        "If unsure, provide your best guess and a confidence score from 0 to 1."
    )

    # Send request to Gemini
    response = prompt_gemini(
        input_prompt=[img, prompt_text],
        schema=SCHEMA,
        temperature=0.0  # deterministic output
    )

    return response


if __name__ == "__main__":
    # Test the system with the example image
    test_image = "images/plant.jpg"

    result = analyze_plant(test_image)

    if result:
        print("\nPlant Analysis Result:")
        print(result)
    else:
        print("No response received from the LLM.")
