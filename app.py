import json
import io as io
import os
import pymongo
import tenacity
import time
import urllib

from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify, send_file
from models import Job, JobList
from openai import AzureOpenAI
from pymongo import UpdateOne, DeleteMany, MongoClient
from tenacity import retry, wait_random_exponential, stop_after_attempt

app = Flask(__name__)

environment = os.getenv('FLASK_ENV')

# Configuration from environment variables
app.config['ENV'] = environment

load_dotenv()

client = AzureOpenAI(
  azure_endpoint=os.getenv("AOAI_ENDPOINT"), 
  api_key=os.getenv("AOAI_KEY"),  
  api_version=os.getenv("AOAI_API_VERSION")
)

# Declare db at the global scope
db = None

# # Set up Azure Speech-to-Text and Text-to-Speech credentials
# speech_key = os.getenv("SPEECH_API_KEY")
# service_region = "eastus"
# speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
# speech_config.speech_synthesis_language = "en-US"
# speech_config.speech_synthesis_voice_name = "en-GB-LibbyNeural"
# speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

# Setup MongoDB Connection
# try:
print("User:", os.environ.get("user"))
print("Password:", os.environ.get("password"))
print("DB Connection String:", os.environ.get("DB_CONNECTION_STRING"))
print("Prod Connection String:", os.environ.get("PROD_CONNECTION_STRING"))

user = os.environ.get("user")
password = os.environ.get("password")
db_url = os.environ.get("DB_CONNECTION_STRING") if environment == "development" else os.environ.get("PROD_CONNECTION_STRING")

if not all([user, password, db_url]):
    raise ValueError("Missing one or more required environment variables: user, password, DB_CONNECTION_STRING/PROD_CONNECTION_STRING")

user = urllib.parse.quote_plus(user)
password = urllib.parse.quote_plus(password)
connection_string = f"mongodb+srv://{user}:{password}@{db_url}"
print(f"Connecting to MongoDB with connection string: {connection_string}")

db_client = pymongo.MongoClient(connection_string)
db = db_client.cosmic_works
print("Successfully connected to MongoDB")
# except pymongo.errors.ConfigurationError as e:
#     print(f"ConfigurationError: {e}")
# except Exception as e:
#     print(f"An error occurred: {e}")

COMPLETIONS_DEPLOYMENT_NAME=os.getenv("COMPLETIONS_DEPLOYMENT_NAME")
EMBEDDINGS_DEPLOYMENT_NAME=os.getenv("EMBEDDINGS_DEPLOYMENT_NAME")

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/send-message', methods=['POST'])
def send_message():
    data = request.get_json()
    user_message = data.get('message', '')

    print(f"Received message: {user_message}")

    try:
        assistant_response = rag_with_vector_search(user_message, 5)
        print(f"Assistant response: {assistant_response}")
        formatted_response = assistant_response
    except Exception as e:
        error_message = f"Error occurred: {str(e)}"
        print(error_message)
        formatted_response = error_message

    return jsonify({'response': formatted_response})

# A system prompt describes the responsibilities, instructions, and persona of the AI.
system_prompt = """
You are a helpful, fun and friendly career advisor for Meander, a tech-focused career management platform. 
Your name is Merlin.
You are designed to answer questions about the job listings on the Meander job board.

Only answer questions related to the information provided in the list of jobs below that are represented
in JSON format.

If you are asked a question that is not in the list, respond with "I don't know."

List of jobs:
"""

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def generate_embeddings(text: str):
    '''
    Generate embeddings from string of text using the deployed Azure OpenAI API embeddings model.
    This will be used to vectorize document data and incoming user messages for a similarity search with
    the vector index.
    '''
    response = client.embeddings.create(input=text, model=EMBEDDINGS_DEPLOYMENT_NAME)
    embeddings = response.data[0].embedding
    time.sleep(0.5) # rest period to avoid rate limiting on AOAI
    return embeddings

def vector_search(collection_name, query, num_results=3):
    """
    Perform a vector search on the specified collection by vectorizing
    the query and searching the vector index for the most similar documents.

    returns a list of the top num_results most similar documents
    """
    global db
    collection = db[collection_name]
    query_embedding = generate_embeddings(query)    
    pipeline = [
        {
            '$search': {
                "cosmosSearch": {
                    "vector": query_embedding,
                    "path": "contentVector",
                    "k": num_results
                },
                "returnStoredSource": True }},
        {'$project': { 'similarityScore': { '$meta': 'searchScore' }, 'document' : '$$ROOT' } }
    ]
    results = collection.aggregate(pipeline)
    return results

def rag_with_vector_search(question: str, num_results: int = 5):
    """
    Use the RAG model to generate a prompt using vector search results based on the
    incoming question.  
    """
    print(f"Received question for vector search: {question}")

    try:
        results = vector_search("jobs", question, num_results=num_results)
        print(f"Vector search results: {results}")

        jobs_list = ""
        for result in results:
            if "contentVector" in result["document"]:
                del result["document"]["contentVector"]
            jobs_list += json.dumps(result["document"], indent=4, default=str) + "\n\n"

        formatted_prompt = system_prompt + jobs_list
        print(f"Formatted prompt: {formatted_prompt}")

        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": question}
        ]

        completion = client.chat.completions.create(messages=messages, model=COMPLETIONS_DEPLOYMENT_NAME)
        print(f"Completion response: {completion}")

        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in RAG with vector search: {str(e)}")
        raise e

if __name__ == '__main__':
    app.run()