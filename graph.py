from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from rag import search_documents, generate_answer


# ============================================
# 1. Define the Graph State
# ============================================

class CarState(TypedDict):
    question: str
    documents: list
    answer: str


# ============================================
# 2. Retrieve Vehicle Documents
# ============================================

def retrieve_documents(state: CarState):

    print("\n[LangGraph] Retrieving documents...")

    question = state["question"]

    documents = search_documents(question)

    return {
        "documents": documents
    }


# ============================================
# 3. Generate Answer with Ollama
# ============================================

def generate_ai_answer(state: CarState):

    print("[LangGraph] Generating answer with Ollama...")

    question = state["question"]

    documents = state["documents"]

    answer = generate_answer(
        question,
        documents
    )

    return {
        "answer": answer
    }


# ============================================
# 4. Build LangGraph
# ============================================

builder = StateGraph(CarState)


# Add nodes

builder.add_node(
    "retrieve",
    retrieve_documents
)

builder.add_node(
    "generate",
    generate_ai_answer
)


# Define workflow

builder.add_edge(
    START,
    "retrieve"
)

builder.add_edge(
    "retrieve",
    "generate"
)

builder.add_edge(
    "generate",
    END
)


# Compile graph

car_graph = builder.compile()


# ============================================
# 5. Test LangGraph
# ============================================

if __name__ == "__main__":

    print("\n===================================")
    print("       USED CAR LANGGRAPH")
    print("===================================")

    question = input(
        "\nEnter your question: "
    )

    result = car_graph.invoke(
        {
            "question": question,
            "documents": [],
            "answer": ""
        }
    )

    print("\n===================================")
    print("             AI ANSWER")
    print("===================================")

    print(result["answer"])

    print("\n===================================")
    print("              SOURCES")
    print("===================================")

    sources = set()

    for document in result["documents"]:

        source = document.metadata.get("source")

        if source:
            sources.add(source)

    for source in sources:

        print(source)