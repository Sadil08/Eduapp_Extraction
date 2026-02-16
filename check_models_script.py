import vertexai
from vertexai.generative_models import GenerativeModel
import os
from dotenv import load_dotenv

load_dotenv()

project_id = os.getenv("GCP_PROJECT_ID")
location = os.getenv("GCP_LOCATION", "asia-south1")

if not project_id:
    print("Error: GCP_PROJECT_ID not found in environment.")
    exit(1)

print(f"Initializing Vertex AI (Project: {project_id}, Location: {location})...")
vertexai.init(project=project_id, location=location)

# List available models is not directly supported in Vertex AI SDK the same way.
# Instead, test a model directly:
try:
    model = GenerativeModel("gemini-2.5-flash")
    response = model.generate_content("Say hello in one word.")
    print(f"gemini-2.5-flash: OK - Response: {response.text}")
except Exception as e:
    print(f"gemini-2.5-flash: FAILED - {e}")

try:
    model = GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Say hello in one word.")
    print(f"gemini-1.5-flash: OK - Response: {response.text}")
except Exception as e:
    print(f"gemini-1.5-flash: FAILED - {e}")
