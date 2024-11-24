from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
import os

app = FastAPI()

API_KEY = "YOUR_OPENAI_APIKEY"
MODEL = "gpt-3.5-turbo"

# Initialize FastEmbed embeddings and LLM
embeddings = FastEmbedEmbeddings()
llm = ChatOpenAI(model=MODEL, temperature=0.3, openai_api_key=API_KEY)

# Initialize vector store
vector_store = Chroma(embedding_function=embeddings)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


class Query(BaseModel):
    question: str

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_path = f"{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load the document
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)

        # Add chunks to the vector store
        vector_store.add_documents(chunks)

        # Clean up the temporary file
        os.remove(file_path)

        return {"message": "Document ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_document(query: Query):
    try:
        # Create a RetrievalQA chain
        retriever = vector_store.as_retriever()
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        # Retrieve the response
        response = chain({"query": query.question})

        # If no relevant documents are found, return a default response
        if not response.get("result") or response["result"].strip() == "":
            return {"response": "No relevant data found in the ingested documents."}

        return {"response": response["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)