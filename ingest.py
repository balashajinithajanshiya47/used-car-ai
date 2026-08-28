from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


DOCUMENTS_PATH = "documents"
VECTORSTORE_PATH = "vectorstore"


print("Loading vehicle documents...")

loader = PyPDFDirectoryLoader(DOCUMENTS_PATH)

documents = loader.load()

print(f"Loaded {len(documents)} pages.")


print("Splitting documents...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


print("Creating FAISS database...")

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

vectorstore.save_local(VECTORSTORE_PATH)

print("FAISS database created successfully!")