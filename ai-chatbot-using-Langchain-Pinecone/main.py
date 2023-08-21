from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder, PromptTemplate)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
import streamlit as st
from streamlit_chat import message
# from utils import *
import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
import PyPDF2
import os

load_dotenv()

# Sidebar contents
with st.sidebar:
    # st.title('LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot. You can ask questions such as:
    - [How can I get an advance?](https://help.toolost.com/hc/en-us/articles/4405003965972-How-do-you-decide-how-big-an-advance-to-give-)
    - [How do I distribute a Cover Song?](https://help.toolost.com/hc/en-us/articles/360054366792-How-do-I-distribute-a-Cover-Song-)
    - [Where does Too Lost deliver my music?](https://help.toolost.com/hc/en-us/articles/360054364032-Where-does-Too-Lost-deliver-my-music-) 
    ''')

st.title("Too Lost Support Agent 🤗💬:")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hi! My name is Navi, how may I help you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

#################

# st.header("Too Lost Support Agent 🤗💬:")
pdf_path = '/Users/jeremyweinapple/Documents/Jobs/TooLost/Projects/LLM/ai-chatbot-using-Langchain-Pinecone/data/articles.pdf'
# pdf_path = 's3://zendesk-chatbot/ai-chatbot-using-Langchain-Pinecone/data/articles.pdf'

# Create a PDF reader object
pdf_reader = PdfReader(pdf_path)

text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200, length_function=len)
chunks = text_splitter.split_text(text=text)

# # embeddings
store_name = 'article_store'
# st.write(f'Loading may take a few seconds :)')

if os.path.exists(f"{store_name}.pkl"):
    with open(f"{store_name}.pkl", "rb") as f:
        VectorStore = pickle.load(f)
    # st.write('Embeddings Loaded from the Disk')
else:
    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    with open(f"{store_name}.pkl", "wb") as f:
        pickle.dump(VectorStore, f)

#################

with textcontainer:
    query = st.text_input("Type any question below: ", key="input")
    if query:
        with st.spinner("typing..."):
            docs = VectorStore.similarity_search(query=query, k=3)

            ########################################
#             Your name is Navi, you are very funny and talk with an exciting tone of voice. Your audience is musicians, so make a pun up about music in each response! 

            prompt_template = """Your name is Navi, your last name is Gator, but its a secret! You are very talkative, but don't use any unnecessary adjectives that may be interpreted incorrectly. you are very funny and talk with an exciting tone of voice. Your audience is musicians, so be really mean!
            Use the following context to answer the human inputted question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            If human inputted question contains text aboung needing documentation, always tell them to 'reach out to support at support@toolost.com'
            {context}

            {chat_history}

            Human: {human_input}
            Chatbot:"""

            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "human_input"])

            memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

            ########################################

            chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=PROMPT, memory = memory)
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, human_input=query)
                # response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
                print(cb)

        # st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
        
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
