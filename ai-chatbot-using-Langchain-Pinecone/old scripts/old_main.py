from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import streamlit as st
from streamlit_chat import message
from utils import *
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
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

### code below isnt working (memory + template)
# system_msg_template = SystemMessagePromptTemplate.from_template(template=
#     "Your name is Found, you are very funny and talkative."
#      "Answer the questions you are being asked and have a sense of humor"
#         "You should be kind and polite in your responses.")

# human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
# prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
# conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

#################

# st.header("Too Lost Support Agent 🤗💬:")
pdf_path = '/Users/jeremyweinapple/Documents/Jobs/TooLost/Projects/LLM/Langchain-PDF-App-using-ChatGPT-main/data/articles.pdf'

# Create a PDF reader object
pdf_reader = PdfReader(pdf_path)

text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
    )
chunks = text_splitter.split_text(text=text)

# # embeddings
store_name = 'article_store'
# st.write(f'Loading may take a few seconds :)')

if os.path.exists(f"{store_name}.pkl"):
    with open(f"{store_name}.pkl", "rb") as f:
        VectorStore = pickle.load(f)
    # st.write('Embeddings Loaded from the Disk')s
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
            # llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            # st.write(response)
    # if query:
    #     with st.spinner("typing..."):
            # conversation_string = get_conversation_string()
            # # st.code(conversation_string)
            # # refined_query = query_refiner(conversation_string, query)
            # # st.subheader("Refined Query:")
            # # st.write(refined_query)
            # # print("refined_query:", refined_query)
            # # context = find_match(refined_query)
            # context = find_match(refined_query)
            # # print(context)  
            # response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

