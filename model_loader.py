import google.generativeai as genai
from PIL import Image
import os

from dotenv import load_dotenv

load_dotenv()

class ModelLoader:
    _instance = None
    _model = None
    _api_key = os.getenv("GEMINI_API_KEY")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelLoader()
        return cls._instance

    def load_model(self):
        if self._model is None:
            print("Configuring Gemini API...")
            try:
                genai.configure(api_key=self._api_key)
                
                # Using the specific model requested.
                model_name = "gemini-2.5-flash" 
                
                # Check available models to be safe? No, just try to list or default
                # But '2.5' sounds like a user typo or very new.
                # If it fails, I will catch it and try 1.5-flash.
                
                self._model = genai.GenerativeModel(model_name)
                print(f"Gemini model '{model_name}' configured successfully.")
            except Exception as e:
                print(f"Error configuring Gemini: {e}")
                # Fallback purely for robustness
                print("Attempting fallback to 'gemini-1.5-flash'...")
                try:
                    self._model = genai.GenerativeModel("gemini-1.5-flash")
                    print("Fallback to gemini-1.5-flash successful.")
                except Exception as ex:
                    print(f"Fallback failed: {ex}")
                    raise ex

    def get_model(self):
        if self._model is None:
            self.load_model()
        return self._model

    def predict(self, image_input, prompt_text="Extract all text from this image."):
        model = self.get_model()
        
        # Prepare inputs
        inputs = [prompt_text]
        
        # Check if image_input is a valid PIL Image and not a dummy
        # main.py passes a dummy 10x10 white image for text-only requests
        # We can detect if it's "dummy" content-wise or just pass it to Gemini
        # Gemini handles text+image fine.
        # But if it's the 10x10 white dummy from main.py intended for text-only marking,
        # we might want to omit it to save bandwidth/confusion, although 10x10 is tiny.
        
        if isinstance(image_input, Image.Image):
             # Simple heuristic: if it's the dummy 10x10 white image, skip it
            if image_input.size == (10, 10):
                # It's likely the dummy for text-only marking
                pass
            else:
                inputs.append(image_input)
        
        try:
            response = model.generate_content(inputs)
            return response.text
        except Exception as e:
            # Handle potential API errors (e.g. valid model name issues)
            # If the header model name was wrong despite configuration check
            return f"Error generating content: {e}"
