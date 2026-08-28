
import os
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================
# FAISS Configuration
# ============================================

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

documents = []
document_names = []


# ============================================
# Read PDF
# ============================================

def read_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================
# Split Text
# ============================================

def split_text(text, chunk_size=500):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# ============================================
# Create FAISS Database
# ============================================

def create_faiss_database(pdf_files):

    global documents
    global document_names

    documents = []
    document_names = []

    for pdf_file in pdf_files:

        text = read_pdf(pdf_file)

        chunks = split_text(text)

        for chunk in chunks:

            if chunk.strip():

                documents.append(chunk)
                document_names.append(
                    os.path.basename(pdf_file)
                )

    if not documents:
        print("No document text found.")
        return None

    # Create embeddings
    embeddings = model.encode(
        documents,
        convert_to_numpy=True
    )

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    print("\n===================================")
    print("       FAISS DATABASE CREATED")
    print("===================================")

    print("Documents:", len(pdf_files))
    print("Text chunks:", len(documents))
    print("Embedding dimension:", dimension)

    return index


# ============================================
# Search FAISS Database
# ============================================

def search_faiss(index, query, top_k=3):

    if index is None:
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for i in indices[0]:

        if i < len(documents):

            results.append({
                "document": document_names[i],
                "text": documents[i]
            })

    return results


# ============================================
# Test FAISS
# ============================================

if __name__ == "__main__":

    print("\n===================================")
    print("       USED CAR FAISS TEST")
    print("===================================")

    pdf_folder = "documents"

    pdf_files = []

    if os.path.exists(pdf_folder):

        for file in os.listdir(pdf_folder):

            if file.lower().endswith(".pdf"):

                pdf_files.append(
                    os.path.join(
                        pdf_folder,
                        file
                    )
                )

    if not pdf_files:

        print("\nNo PDF files found.")

        print(
            "\nCreate a 'documents' folder and "
            "place your vehicle PDF files inside it."
        )

    else:

        index = create_faiss_database(
            pdf_files
        )

        print("\nSearching for vehicle information...")

        results = search_faiss(
            index,
            "What is the insurance claim history?",
            top_k=3
        )

        print("\n===================================")
        print("           SEARCH RESULTS")
        print("===================================")

        for result in results:

            print("\nDocument:")
            print(result["document"])

            print("\nInformation:")
            print(result["text"])

            print("-----------------------------------")

