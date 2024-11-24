# FastAPI Application for Document Ingestion and Querying

This FastAPI application allows you to ingest PDF documents, store their content in a vector database, and query the documents using OpenAI's GPT-3.5-turbo model for relevant answers.

---

## Features

- **Document Ingestion**: Upload and process PDF documents for storage in a vector database.
- **Query System**: Ask questions about ingested documents and receive answers based on their content.
- **Fallback Handling**: Provides a default response if no relevant data is found.

---

```
API_KEY = "YOUR_OPENAI_APIKEY"
MODEL = "gpt-3.5-turbo"

# Initialize embeddings and language model
embeddings = FastEmbedEmbeddings()
llm = ChatOpenAI(model=MODEL, temperature=0.3, openai_api_key=API_KEY)

# Initialize vector store
vector_store = Chroma(embedding_function=embeddings)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

---

## Running the Application

```
fastapi run main.py
```

## feel free to use this code in your apps its pretty simple and hope it helps your application