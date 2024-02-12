# import os
# from langchain_community.embeddings import GooglePalmEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain.chains import ConversationalRetrievalChain

# from langchain_community.llms.google_palm import GooglePalm
# llm = GooglePalm(google_api_key=os.environ['GOOGLE_API_KEY'])
# llm.temperature = 0.1
import os
google_api_key=os.environ.get('GOOGLE_API_KEY')
print(google_api_key)
from langchain_google_genai import GoogleGenerativeAI
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=google_api_key)
llm.temperature = 0.1
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('nlp.pdf')
documents = loader.load()

# Split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

embeddings = GooglePalmEmbeddings(google_api_key=google_api_key)
vectorstore = Chroma.from_documents(documents, embeddings)
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(),verbose=False,return_source_documents=True)

query="What is NLP?"

chat_history=[]
result =  qa_chain.invoke({"question": query, "chat_history": chat_history})
print(f"Answer: " + result["answer"])
chat_history.append((query, result["answer"]))