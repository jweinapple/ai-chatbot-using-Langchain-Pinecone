from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key='13d3c4ff-e4cc-4717-9432-52eb17f0f475', # find at app.pinecone.io
              environment='us-west1-gcp-free' # next to api key in console
             )
index = pinecone.Index('support-chatbot') # index name from pinecone)


# def find_match(input):
#     input_em = model.encode(input).tolist()
#     result = index.query(input_em, top_k=2, includeMetadata=True)
    # return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

# chat GPT answer:
def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    matches = result.get('matches', [])

    if len(matches) >= 2:
        return matches[0]['metadata']['text'] + "\n" + matches[1]['metadata']['text']
    elif len(matches) == 1:
        return matches[0]['metadata']['text']
    else:
        return "No matches found."


def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
