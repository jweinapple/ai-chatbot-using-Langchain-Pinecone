# Chatbot Answering from Your Own Knowledge Base: Langchain, ChatGPT, Pinecone, and Streamlit

## Deployment

#### 3. To install require packages 

```bash
  pip install -r requirements.txt
```
#### 4. Replace your own document in **data** folder

#### 5. Replace your own OpenAI, Pinecone API Key and Pinecone environment in indexing.py, main.py & utils.py
 - [OpenAI API Key](https://platform.openai.com)
 - [Pinecone](app.pinecone.io)
   - When you are creating the pinecone index make sure, **Dimensions of the index is 384**
 
#### 6. Run the web app
```bash
  streamlit run main.py
```
