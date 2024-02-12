"""modules imported"""
import os
from database import chroma_instances
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAI
google_api_key=os.environ.get('GOOGLE_API_KEY')
print(google_api_key)
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=google_api_key)
llm.temperature = 0.1

chat_history=[]

app = FastAPI()
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), pdf_name: str = None):
    """Function to upload a file"""
    pdf_name = pdf_name[:-4]
    print(pdf_name)
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
        embeddings = GooglePalmEmbeddings(google_api_key=google_api_key)
        # Create the open-source embedding function
        #embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db_dir = os.path.join("", "ChromaDB")
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, f"{pdf_name or file.filename}.db")
        db = Chroma.from_documents(docs, embeddings, collection_name=pdf_name or file.filename, persist_directory=db_path)
        # Store the Chroma instance in the dictionary
        chroma_instances[file.filename] = db
        print(chroma_instances)

        # Optionally, you can return some information
        return {"message": f"File '{file.filename}' uploaded successfully and processed with collection name '{pdf_name or file.filename}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    
@app.get("/query/")
async def query_pdf(pdf_name: str, question: str):
    """Query the function version."""
    try:
        # Check if the specified PDF exists in the dictionary
        
        if pdf_name not in chroma_instances:
            raise HTTPException(status_code=404, detail=f"PDF '{pdf_name}' not found in the database")
        print(chroma_instances)
        print(chroma_instances[pdf_name])
        # Retrieve the Chroma instance for the specified PDF
        db = chroma_instances[pdf_name]

        # Perform similarity search
        #docs = db.similarity_search(question)

        #PaLM 2 Integration

        qa_chain = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(),verbose=False,return_source_documents=True)
        result =  qa_chain.invoke({"question": question, "chat_history": chat_history})
        print("Answer: " + result["answer"])
        chat_history.append((question, result["answer"]))
        # Return the content of the most relevant document
        if result:
            return {"answer": result["answer"]}
        else:
            return {"answer": "No relevant information found."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
