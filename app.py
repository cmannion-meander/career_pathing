import json
import io as io
import networkx as nx
import os
import pymongo
import tenacity
import time
import urllib

from collections import defaultdict
from datetime import datetime
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
        # print(f"Vector search results: {results}")

        jobs_list = ""
        for result in results:
            if "contentVector" in result["document"]:
                del result["document"]["contentVector"]
            jobs_list += json.dumps(result["document"], indent=4, default=str) + "\n\n"

        formatted_prompt = system_prompt + jobs_list
        # print(f"Formatted prompt: {formatted_prompt}")

        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": question}
        ]

        completion = client.chat.completions.create(messages=messages, model=COMPLETIONS_DEPLOYMENT_NAME)
        # print(f"Completion response: {completion}")

        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in RAG with vector search: {str(e)}")
        raise e

def recommend_based_on_skills(skills: str, num_results: int = 5):
    question = f"Find jobs that match these skills: {skills}"
    return rag_with_vector_search(question, num_results)


def recommend_based_on_experience(experience: str, num_results: int = 5):
    question = f"Find jobs that match this experience: {experience}"
    return rag_with_vector_search(question, num_results)

def recommend_based_on_location(location: str, num_results: int = 5):
    question = f"Find jobs available in {location}"
    return rag_with_vector_search(question, num_results)


def vector_search_profiles(collection_name, query, num_results=3):
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
                "returnStoredSource": True
            }
        },
        {
            '$project': {
                'similarityScore': { '$meta': 'searchScore' },
                'document': '$$ROOT'
            }
        }
    ]
    results = list(collection.aggregate(pipeline))
    
    return [result['document'] for result in results]

def print_profile_search_result(result):
    '''
    Print the search result document in a readable format.
    '''
    document = result['document']
    print(f"Similarity Score: {result['similarityScore']:.4f}")
    
    # Assuming the profiles have these fields
    print(f"Profile ID: {document['profileId']}")
    print(f"Path: {document.get('path', 'N/A')}")
    print(f"Location: {document.get('location', 'N/A')}")
    print(f"Years of Experience: {document.get('yearsOfExp', 'N/A')}")
    print(f"Total Promotions: {document.get('totalPromotions', 'N/A')}")
    print(f"Score: {document.get('score', 'N/A')}")
    print("Roles:")
    
    for role in document['roles']:
        print(f"  - Title: {role['title']}")
        print(f"    Company: {role.get('company', 'Unknown')}")
        print(f"    Start Date: {role.get('startDate', 'N/A')}")
        print(f"    End Date: {role.get('endDate', 'Present')}")
        print(f"    Years in Role: {role.get('yearsInRole', 'N/A')}")
        print(f"    Location: {role.get('location', 'N/A')}")
        print("")
    print("")

def build_career_path_graph(profiles):
    """
    Build a career path graph based on the provided profiles.

    Args:
    - profiles: A list of dictionaries representing profile data or a JSON string.

    Returns:
    - A directed graph representing career transitions.
    """
    # If profiles is a string, convert it to a list of dictionaries
    if isinstance(profiles, str):
        try:
            profiles = json.loads(profiles)
            print("Profiles successfully converted from JSON string to list of dictionaries.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    
    # Verify that profiles is now a list of dictionaries
    if not isinstance(profiles, list):
        print("Error: Profiles should be a list of dictionaries.")
        return None
    
    for profile in profiles:
        if not isinstance(profile, dict):
            print("Error: Each profile should be a dictionary.")
            return None
    
    # Debug: Print the structure of the profiles
    print("Profiles structure:")
    for profile in profiles:
        print(json.dumps(profile, indent=4))

    transitions = defaultdict(lambda: defaultdict(list))

    for profile in profiles:
        roles = profile.get('roles', [])
        for i in range(len(roles) - 1):
            from_role = roles[i]
            to_role = roles[i + 1]
            transitions[from_role['title']][to_role['title']].append({
                "years_in_role": from_role.get('years_in_role', 0),
                "industry": from_role.get('industry', 'N/A'),
                "location": from_role.get('location', 'N/A'),
                "profile_id": profile.get('profile_id', 'N/A')
            })

    aggregated_transitions = []
    for from_role, to_roles in transitions.items():
        for to_role, details in to_roles.items():
            average_years = sum(item['years_in_role'] for item in details) / len(details)
            industries = list(set(item['industry'] for item in details))
            locations = list(set(item['location'] for item in details))
            profile_ids = [item['profile_id'] for item in details]
            aggregated_transitions.append({
                "from_role": from_role,
                "to_role": to_role,
                "average_years_in_role": average_years,
                "profile_ids": profile_ids,
                "industries": industries,
                "locations": locations
            })

    G = nx.DiGraph()
    for transition in aggregated_transitions:
        G.add_edge(
            transition['from_role'],
            transition['to_role'],
            weight=transition['average_years_in_role'],
            transition_count=len(transition['profile_ids']),
            industries=transition['industries'],
            locations=transition['locations']
        )
    
    # Debug: Print the graph edges
    print("Graph edges:")
    for edge in G.edges(data=True):
        print(edge)
    
    return G

def recommend_next_role(current_role, profiles):
    career_path_graph = build_career_path_graph(profiles)
    print("Career Path created\n\n")
    instructional_prompt = f"""
        You are Merlin, a helpful, fun, and friendly career advisor for Meander, a tech-focused career management platform.
        You are making data-driven recommendations to help the user decide on what role they should consider next based on which roles top
        performers have taken. Where possible, you provide percentage breakdowns of which roles people have picked. 
        
        The user is currently in a {current_role} role.
        
        Here is the career path graph based on our dataset: \n\n {career_path_graph}.
        """

    try:
        recommended_response = client.chat.completions.create(
            model=COMPLETIONS_DEPLOYMENT_NAME,
            messages = [
                {"role": "system", "content": instructional_prompt},
                {"role": "user", "content": "What career moves would you recommend next?"}
            ]
        )  # get a new response from GPT where it can see the function response

        # print(f"Second response: {second_response} \n\n\n")

        formatted_response = recommended_response.choices[0].message.content
        return formatted_response
    except:
        return "Unable to recommend a next role"
    # if current_role in career_path_graph:
    #     recommendations = sorted(career_path_graph[current_role].items(), key=lambda x: x[1]['weight'])
    #     return [
    #         {
    #             "role": role,
    #             "average_years_in_role": data['weight'],
    #             "transition_count": data['transition_count'],
    #             "industries": data['industries'],
    #             "locations": data['locations']
    #         }
    #         for role, data in recommendations
    #     ]
    # else:
    #     return []

def career_path_advisor(current_role: str):
    query = f"What {current_role} profiles do you have?"
    print(f"Query: {query}\n")
    profiles = vector_search_profiles("profiles", query, num_results=5)

    profile_json = generate_profile_json(profiles)

    print(f"Profiles found:\n {profile_json}\n\n")

    recommendations = recommend_next_role(current_role, profile_json)
    if not recommendations:
        return "No recommendations available for the specified role."
    return f"Based on our analysis, here are the recommended next roles for a {current_role}:\n\n{recommendations}"
    
    # recommendations_str = "\n".join([
    #     f"{i + 1}. {rec['role']} (average time in role: {rec['average_years_in_role']:.2f} years, "
    #     f"common industries: {', '.join(rec['industries'])}, common locations: {', '.join(rec['locations'])})"
    #     for i, rec in enumerate(recommendations)
    # ])
    # return f"Based on our analysis, here are the recommended next roles for a {current_role}:\n\n{recommendations_str}"

def identify_role_sources(target_role):
    query = f"What {target_role} profiles do you have?"
    print(f"Query: {query}\n")
    profiles = vector_search_profiles("profiles", query, num_results=5)

    profile_json = generate_profile_json(profiles)

    print(f"Profiles found:\n {profile_json}\n\n")

    career_path_graph = build_career_path_graph(profile_json)
    
    print("Career Path created\n\n")
    instructional_prompt = f"""
        You are Merlin, a helpful, fun, and friendly career advisor for Meander, a tech-focused career management platform.
        You are making data-driven recommendations to help the user understand what roles typically transition into target roles. Where possible, you provide percentage breakdowns of which roles people have come from. 
        
        The user is currently is interested in paths to a {target_role} role.
        
        Here is the career path graph based on our dataset: \n\n {career_path_graph}.
        """

    try:
        recommended_response = client.chat.completions.create(
            model=COMPLETIONS_DEPLOYMENT_NAME,
            messages = [
                {"role": "system", "content": instructional_prompt},
                {"role": "user", "content": f"What pathways lead to {target_role}?"}
            ]
        )  # get a new response from GPT where it can see the function response

        # print(f"Second response: {second_response} \n\n\n")

        formatted_response = recommended_response.choices[0].message.content
        return formatted_response
    except:
        return "Unable to identify role pathways"

def generate_profile_json(profiles):
    enhanced_profiles = []
    for profile in profiles:
        profile_dict = {
            "profile_id": profile['profileId'],
            "path": profile.get('path', 'Unknown'),  # Assuming 'path' might be different for each profile
            "roles": []
        }
        for role in profile.get('roles', []):
            start_date = role.get('startDate')
            end_date = role.get('endDate', 'Present')
            years_in_role = 0  # Default to 0 if dates are not valid

            if start_date:
                start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
                if end_date and end_date != 'Present':
                    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    years_in_role = (end_date_dt - start_date_dt).days / 365
                else:
                    end_date_dt = datetime.now()
                    years_in_role = (end_date_dt - start_date_dt).days / 365

            role_dict = {
                "title": role['title'],
                "company": role['company'],
                "start_date": start_date,
                "end_date": end_date,
                "years_in_role": years_in_role,
                "industry": role.get('industry', 'N/A'),
                "location": role.get('location', 'N/A')
            }
            profile_dict["roles"].append(role_dict)
        enhanced_profiles.append(profile_dict)
    
    return json.dumps(enhanced_profiles, indent=4)


def job_search_advisor():
    prompt = """
    The 7-Step Job Search Framework
    This framework's goal is to help you navigate the job search process quickly and effectively. We start by defining precisely what you're looking for, then create a structured approach to help you land the perfect role.
    
    Step 1: Clarify what values and attributes are most important
    Begin with introspection to clarify what you truly seek in your next role. Focus on your core values and what aspects of a job are most fulfilling to you. Make a list of desirable job attributes (e.g., salary, culture, management) and define what "good" and "bad" look like for each to clarify your search criteria.
    
    Step 2: Create a shortlist of target companies and types of role
    Use your clarified values and criteria to identify 15 companies that align with your professional aspirations and values. Conduct thorough research on these companies to ensure they meet your "good" criteria, allowing for a focused yet broad enough search to increase your chances of success.
    
    Step 3: Implement strategic networking and relationship-building
    Identify and try to connect with 15 key individuals within each target company who can influence hiring decisions. Develop engagement plans for each contact, utilizing LinkedIn and other resources to find common ground and ways to add value, aiming to transition these connections into meaningful relationships.
    
    Step 4: Develop deep company insight and role understanding
    Maximize your relationships to gain insights into the company's goals, challenges, and initiatives. Use this information to tailor your application and narrative and prepare for interviews, ensuring you're seen as a candidate who deeply understands and aligns with the company's needs.
    
    Step 5: Craft your personal narrative
    Prepare your narrative for applications and interviews, focusing on how your skills, experiences, and values align with the company's goals and challenges. Utilize a structured approach to craft compelling stories that demonstrate your potential impact and place the company's needs at the forefront.
    
    Step 6: Interview preparation and value demonstration
    Beyond traditional interview preparation, create a Value Validation Project (VVP, H/T Austin Belcak) for each target company. This pitch deck should outline the company's significant challenges or opportunities, propose solutions, and highlight why you are uniquely positioned to execute these plans.
    
    Step 7: Continuous refinement and persistence
    Reflect on each interaction and interview to refine your approach continually. Persistence is vital; learn from each experience, adjust your strategies as needed, and focus on your selected companies and roles.
    
    Please provide specific advice based on this framework to help job seekers navigate their job search.
    """
    
    return prompt

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
                        "description": "The user's preferred job type",
                    },
                },
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_based_on_skills",
            "description": "Find roles that match user's skills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skills": {
                        "type": "string",
                        "description": "The user's skills to match jobs",
                    },
                },
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_based_on_experience",
            "description": "Find roles that match user's experience.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experience": {
                        "type": "string",
                        "description": "The user's experience to match jobs",
                    },
                },
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_based_on_location",
            "description": "Find roles available in a specific location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to find jobs",
                    },
                },
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "career_path_advisor",
            "description": "Recommend next career steps based on current role.",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_role": {
                        "type": "string",
                        "description": "The user's current role to base recommendations on",
                    },
                },
            },
        }
    },
        {
        "type": "function",
        "function": {
            "name": "identify_role_sources",
            "description": "Recommend pathways into a target role.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_role": {
                        "type": "string",
                        "description": "The role a user wants to enter.",
                    },
                },
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "job_search_advisor",
            "description": "Guide job seekers through their search strategy using the 7-Step Job Search Framework.",
            "parameters": {}
        }
    }
]

available_functions = {
    "rag_with_vector_search": rag_with_vector_search,
    "recommend_based_on_skills": recommend_based_on_skills,
    "recommend_based_on_experience": recommend_based_on_experience,
    "recommend_based_on_location": recommend_based_on_location,
    "career_path_advisor": career_path_advisor,
    "identify_role_sources": identify_role_sources,
    "job_search_advisor": job_search_advisor,
}

@app.route('/send-message', methods=['POST'])
def send_message():
    answers = []
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

    # print(f"Step 1 response message: {response_message}")

    tool_calls = response_message.tool_calls

    # Step 2: check if GPT wanted to call a function
    if tool_calls:
        # print(f"Tool call: {tool_calls}")
        # Step 3: call the function
        messages.append(response_message)
        
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            
            try:
                function_response = function_to_call(**function_args)                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })
                answers.append(function_response)
            except Exception as e:
                print(f"Error calling function {function_name}: {e}")
            
            # print(f"\n\n Messages after appending function response: {messages} \n\n")


        # Create an instructional prompt for the second GPT completion call
        instructional_prompt = f"""
        You are Merlin, a helpful, fun, and friendly career advisor for Meander, a tech-focused career management platform.
        You have received the following information. Please present this information in a 
        natural, friendly, and easy-to-understand manner for the user.
        
        Here is the question you were asked: {user_message}.

        Here are the relevant answers to the question: {answers}
        """

        try:
            second_response = client.chat.completions.create(
                model=COMPLETIONS_DEPLOYMENT_NAME,
                messages = [
                    {"role": "system", "content": instructional_prompt},
                    {"role": "user", "content": user_message}
                ]
            )  # get a new response from GPT where it can see the function response

            # print(f"Second response: {second_response} \n\n\n")

            formatted_response = second_response.choices[0].message.content
            # print(f"Step 4 response message: {formatted_response} \n\n\n")

            return jsonify({'response': formatted_response})
        except Exception as e:
            print(f"Error getting second response: {e}")
            return jsonify({'error': str(e)})

    else:
        # Extract descriptions from the tools list
        function_descriptions = [tool["function"]["description"] for tool in tools]
        
        # Create a user-friendly list of functions
        function_list = "\n".join([f"- {desc}" for desc in function_descriptions])
        
        # Generate the apology and guidance message
        apology_response = client.chat.completions.create(
            model=COMPLETIONS_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides information about available functions and helps to apologize when something goes wrong."},
                {"role": "user", "content": f"""Can you provide a friendly apology and guide the user on how to use the available functions? Make sure that the output is succinct and targeted at the user. Here are the available functions: \n\n{function_list}"""}
            ]
        )
        apology_message = apology_response.choices[0].message.content

        return jsonify({'response': apology_message})

@app.route('/convert-resume', methods=['POST'])
def convert_resume():
    military_resume = request.form['resume_text']
    prompt = f"Here is a military resume:\n{military_resume}\n\nConvert this into a civilianized resume suitable for a corporate job application:"
    
    response = client.chat.completions.create(
        model=COMPLETIONS_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "Assistant is an expert recruiter trained to convert military resumes into civilian format."},
            {"role": "user", "content": prompt}
        ]
    )
    
    civilian_resume = response.choices[0].message.content if response.choices else "Conversion failed. Please try again."

    # Render both the original and converted resumes
    return render_template('resume.html', original=military_resume, response=civilian_resume)

def vector_search_jobs(question, num_results=5):
    # Call the vector search function
    results = vector_search("jobs", question, num_results=num_results)
    
    jobs = []
    for result in results:
        job_data = result["document"]
        if "contentVector" in job_data:
            del job_data["contentVector"]
        jobs.append(job_data)
    
    return jobs

@app.route('/search-jobs', methods=['GET', 'POST'])
def search_jobs():
    if request.method == 'POST':
        question = request.form['question']
        jobs = vector_search_jobs(question)  # Call the function to search jobs
        return render_template('job_results.html', jobs=jobs, question=question)
    return render_template('search_jobs.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/career_chat')
def career_chat():
    return render_template('index.html')

@app.route('/resume_builder')
def resume_builder():
    return render_template('resume.html')

@app.route('/resources')
def resources():
    return render_template('resources.html')

if __name__ == '__main__':
    app.run()