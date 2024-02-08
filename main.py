from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os

app = FastAPI()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), pdf_name: str = None):
    try:
        # Check if the file is a PDF
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Save the uploaded PDF file
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())

        # Load the PDF document
        loader = PyPDFLoader(file.filename)
        documents = loader.load()

        # Split it into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # Create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db_dir = os.path.join("", "ChromaDB")
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, f"{pdf_name or file.filename}.db")
        db = Chroma.from_documents(docs, embedding_function, collection_name=pdf_name or file.filename, persist_directory=db_path)
        # query it
        query = "What did Natural Language Processing?"
        docs = db.similarity_search(query)

        # print results
        print(docs[0].page_content)
        # Optionally, you can return some information
        return {"message": f"File '{file.filename}' uploaded successfully and processed with collection name '{pdf_name or file.filename}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))