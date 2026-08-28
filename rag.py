from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


# ============================================
# Configuration
# ============================================

VECTORSTORE_PATH = "vectorstore"
OLLAMA_MODEL = "llama3.2:latest"


# ============================================
# 1. Load Embedding Model
# ============================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================
# 2. Load FAISS Database
# ============================================

print("Loading FAISS database...")

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)


# ============================================
# 3. Create Retriever
# ============================================

print("Creating retriever...")

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================
# 4. Connect to Ollama
# ============================================

print("Connecting to Ollama...")

llm = OllamaLLM(
    model=OLLAMA_MODEL
)


# ============================================
# 5. Search Documents
# ============================================

def search_documents(question):

    documents = retriever.invoke(question)

    return documents


# ============================================
# 6. Generate AI Answer
# ============================================

def generate_answer(question, documents):

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a Used Car AI Assistant.

Your job is to answer questions about a used vehicle
using ONLY the information provided in the vehicle documents.

IMPORTANT RULES:

1. Use only the provided documents.
2. Do not invent information.
3. If the answer is not present in the documents,
   say that the information is not available.
4. Give a clear and simple answer.
5. Mention important dates, mileage, repairs, claims,
   or problems when relevant.

VEHICLE DOCUMENTS:
------------------

{context}

USER QUESTION:
--------------

{question}

ANSWER:
"""

    answer = llm.invoke(prompt)

    return answer


# ============================================
# 7. Main Application
# ============================================

if __name__ == "__main__":

    print("\n===================================")
    print("       USED CAR AI ASSISTANT")
    print("===================================")

    question = input(
        "\nEnter your question: "
    )

    print("\nSearching vehicle documents...")

    documents = search_documents(question)

    if not documents:

        print("\nNo relevant documents found.")

    else:

        print(
            f"\nFound {len(documents)} relevant documents."
        )

        print("\nGenerating AI answer...")

        answer = generate_answer(
            question,
            documents
        )

        print("\n===================================")
        print("             AI ANSWER")
        print("===================================")

        print(answer)

        print("\n===================================")
        print("              SOURCES")
        print("===================================")

        sources = set()

        for document in documents:

            source = document.metadata.get("source")

            if source:
                sources.add(source)

        for source in sources:

            print(source)