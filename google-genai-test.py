import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

response = client.models.generate_content(model="gemini-2.0-flash", contents="Hello, world!")
print(response.text)