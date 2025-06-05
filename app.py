from flask import Flask, render_template,jsonify,request
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAI as genai
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core. prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

#initialize Flask app
app=Flask(__name__)

load_dotenv()
print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = download_huggingface_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever=docsearch.as_retriever(search_type="similarity",search_kwargs={"k": 3})

model = genai(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain=create_stuff_documents_chain(model,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)


@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    reponse = rag_chain.invoke({"input": msg})
    print("Response:", reponse['answer'])
    return str(reponse['answer'])


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=8080)