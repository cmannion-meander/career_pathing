import azure.cognitiveservices.speech as speechsdk
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

# Configuration from environment variables
app.config['ENV'] = os.getenv('FLASK_ENV')

load_dotenv()

client = AzureOpenAI(
  azure_endpoint=os.getenv("AOAI_ENDPOINT"), 
  api_key=os.getenv("AOAI_KEY"),  
  api_version=os.getenv("AOAI_API_VERSION")
)

# Set up Azure Speech-to-Text and Text-to-Speech credentials
speech_key = os.getenv("SPEECH_API_KEY")
service_region = "eastus"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_synthesis_language = "en-US"
speech_config.speech_synthesis_voice_name = "en-GB-LibbyNeural"
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

# Setup Mongo DB Connection
user = urllib.parse.quote_plus(os.environ.get("user"))
password = urllib.parse.quote_plus(os.environ.get("password"))
DB_URL = os.environ.get("DB_CONNECTION_STRING")
CONNECTION_STRING = f"mongodb+srv://{user}:{password}@{DB_URL}"
db_client = pymongo.MongoClient(CONNECTION_STRING)
db = db_client.cosmic_works

COMPLETIONS_DEPLOYMENT_NAME=os.getenv("COMPLETIONS_DEPLOYMENT_NAME")
EMBEDDINGS_DEPLOYMENT_NAME=os.getenv("EMBEDDINGS_DEPLOYMENT_NAME")

@app.route('/', methods=['GET'])
def home():
    # Just render the initial form
    return render_template('index.html')

@app.route('/send-message', methods=['POST'])
def send_message():
    data = request.get_json()
    user_message = data['message']
    try:
        assistant_response = rag_with_vector_search(user_message, 5)

        # Add line breaks after punctuation
        # formatted_response = re.sub(r'(?<=[.!?])\s', '\n\n', assistant_response)
        formatted_response = assistant_response
    except Exception as e:
        formatted_response = "There was an error processing your request."
    
    return jsonify({'response': formatted_response})

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech_route():
    print("Received text-to-speech POST request")
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        print("No text provided")
        return jsonify({'error': 'No text provided'}), 400

    audio_data = text_to_speech(text)
    if audio_data:
        print("Returning synthesized audio")
        # Convert the byte stream to a response
        return send_file(
            io.BytesIO(audio_data),
            mimetype="audio/wav",
            as_attachment=False,  # Adjust based on whether you want the file to be downloaded or played
        )
    else:
        print("Failed to synthesize speech")
        return jsonify({'error': 'Failed to synthesize speech'}), 500

def text_to_speech(text):
    """Convert text to speech using Azure Cognitive Services."""
    try:
        print(f"Converting text to speech: {text}")
        # Request speech synthesis
        result = speech_synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Text-to-speech conversion successful.")
            return result.audio_data  # Returning audio data
        else:
            print(f"Error synthesizing audio: {result.reason}")
            return None
    except Exception as ex:
        print(f"Error in text-to-speech conversion: {ex}")
        return None



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
    # perform the vector search and build product list
    results = vector_search("jobs", question, num_results=num_results)
    jobs_list = ""
    for result in results:
        if "contentVector" in result["document"]:
            del result["document"]["contentVector"]
        jobs_list += json.dumps(result["document"], indent=4, default=str) + "\n\n"

    # generate prompt for the LLM with vector results
    formatted_prompt = system_prompt + jobs_list

    # prepare the LLM request
    messages = [
        {"role": "system", "content": formatted_prompt},
        {"role": "user", "content": question}
    ]

    completion = client.chat.completions.create(messages=messages, model=COMPLETIONS_DEPLOYMENT_NAME)
    return completion.choices[0].message.content

if __name__ == '__main__':
    app.run()
