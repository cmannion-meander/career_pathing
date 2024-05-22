import json
import io as io
import os
import pymongo
import tenacity
import time
import urllib

from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify, send_file
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
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

### Career advice prompts

def rag_with_vector_search(question: str, num_results: int = 3):
    """
    Use the RAG model to generate a prompt using vector search results based on the
    incoming question.  
    """
    print(f"Received question for vector search: {question}")

    system_prompt = """
    You are a helpful, fun and friendly career advisor for Meander, a tech-focused career management platform. 
    Your name is Merlin.
    You are designed to answer questions about the job listings on the Meander job board.

    Only answer questions related to the information provided in the list of jobs below that are represented
    in JSON format.

    If you are asked a question that is not in the list, respond with "I don't know."

    List of jobs:
    """

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

tools = [
    {
        "type": "function",
        "function": {
            "name": "rag_with_vector_search",
            "description": "Find roles that match user preferences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The users question to be answered",
                    },
                },
            },
        }

    }
]

available_functions = {
    "rag_with_vector_search": rag_with_vector_search
}

@app.route('/send-message', methods=['POST'])
def send_message():
    data = request.get_json()
    user_message = data.get('message', '')

    print(f"Received message: {user_message}")
   # Step 1: send the conversation and available functions to GPT
    messages = [{"role": "user", "content": user_message}]
    
    response = client.chat.completions.create(
        messages=messages, 
        model=COMPLETIONS_DEPLOYMENT_NAME,
        tools=tools,
        tool_choice="auto", )
    
    response_message = response.choices[0].message

    tool_calls = response_message.tool_calls

    # Step 2: check if GPT wanted to call a function
    if tool_calls:
        # Step 3: call the function
        messages.append(response_message) 
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            # Convert the function response to a JSON string if it's a dictionary
            if isinstance(function_response, dict):
                function_response = json.dumps(function_response)

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            ) 
        second_response = client.chat.completions.create(
            model=COMPLETIONS_DEPLOYMENT_NAME,
            messages=messages,
        )  # get a new response from GPT where it can see the function response

        formatted_response = second_response.choices[0].message.content

        return jsonify({'response': formatted_response})

    else:
        # Generating an apology and help message
        apology_response = client.chat.completions.create(
            model=COMPLETIONS_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides information about available functions and helps to apologize when something goes wrong."},
                {"role": "user", "content": f"""Can you provide a friendly apology and guide the user on how to use the available functions listed in {tools}? Make sure that the output is succint and suitable for a Slack channel."""}
            ]
        )
        apology_message = apology_response.choices[0].message.content
               
        return jsonify({'response': apology_message})

if __name__ == '__main__':
    app.run()