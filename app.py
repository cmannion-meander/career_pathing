import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

chatClient = AzureOpenAI(
  azure_endpoint=os.getenv("AOAI_ENDPOINT"), 
  api_key=os.getenv("AOAI_KEY"),  
  api_version="2023-05-15"
)

chatResponse = chatClient.chat.completions.create(
    model="mannion-gpt35",
    messages=[
        {"role": "system", "content": "You are a helpful, fun and friendly sales assistant for Cosmic Works, a bicycle and bicycle accessories store."},
        {"role": "user", "content": "Do you sell bicycles?"},
        {"role": "assistant", "content": "Yes, we do sell bicycles. What kind of bicycle are you looking for?"},
        {"role": "user", "content": "I'm not sure what I'm looking for. Could you help me decide?"}
    ]
)

print(chatResponse.choices[0].message.content)