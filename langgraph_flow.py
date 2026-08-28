from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from faiss_db import create_faiss_database, search_faiss
from crew import analyze_vehicle

import os


# ============================================
# LANGGRAPH STATE
# ============================================

class VehicleState(TypedDict):
    pdf_files: list
    vehicle_information: str
    reasoning: str
    final_result: str


# ============================================
# 1. LOAD PDF DOCUMENTS
# ============================================

def load_documents(state: VehicleState):

    print("\n===================================")
    print("       LANGGRAPH: LOAD DOCUMENTS")
    print("===================================")

    folder = "documents"

    pdf_files = []

    if os.path.exists(folder):

        for file in os.listdir(folder):

            if file.lower().endswith(".pdf"):

                pdf_files.append(
                    os.path.join(folder, file)
                )

    print("PDF files found:", len(pdf_files))

    return {
        "pdf_files": pdf_files
    }


# ============================================
# 2. RETRIEVE INFORMATION FROM FAISS
# ============================================

def retrieve_information(state: VehicleState):

    print("\n===================================")
    print("       LANGGRAPH: FAISS SEARCH")
    print("===================================")

    pdf_files = state["pdf_files"]

    index = create_faiss_database(pdf_files)

    # Search separately
    insurance = search_faiss(
        index,
        "insurance claim accident rear bumper damage",
        top_k=1
    )

    service = search_faiss(
        index,
        "service maintenance repair brake transmission engine sensor",
        top_k=1
    )

    inspection = search_faiss(
        index,
        "inspection engine transmission brakes tyres body warning light",
        top_k=1
    )

    vehicle_information = ""

    # ----------------------------------------
    # Insurance
    # ----------------------------------------

    vehicle_information += "\nINSURANCE HISTORY:\n"

    if insurance:
        vehicle_information += insurance[0]["text"]

    # ----------------------------------------
    # Service
    # ----------------------------------------

    vehicle_information += "\n\nSERVICE HISTORY:\n"

    if service:
        vehicle_information += service[0]["text"]

    # ----------------------------------------
    # Inspection
    # ----------------------------------------

    vehicle_information += "\n\nINSPECTION REPORT:\n"

    if inspection:
        vehicle_information += inspection[0]["text"]

    print("FAISS retrieval completed.")

    return {
        "vehicle_information": vehicle_information
    }


# ============================================
# 3. REACT-STYLE REASONING
# ============================================

def reasoning_node(state: VehicleState):

    print("\n===================================")
    print("       LANGGRAPH: REASONING")
    print("===================================")

    information = state["vehicle_information"]

    reasoning = f"""
Analyze this vehicle information before the
CrewAI analysts process it.

Identify:

- Important insurance events
- Important service events
- Inspection concerns
- Positive points
- Items that should be checked before purchase

Use ONLY the information provided.

Do not add assumptions.

VEHICLE INFORMATION:

{information}
"""

    print("Reasoning stage completed.")

    return {
        "reasoning": reasoning
    }


# ============================================
# 4. CREWAI ANALYSIS
# ============================================

def crewai_node(state: VehicleState):

    print("\n===================================")
    print("       LANGGRAPH: CREWAI")
    print("===================================")

    information = state["vehicle_information"]

    reasoning = state["reasoning"]

    # Keep the information organized
    crew_input = f"""
VEHICLE INFORMATION:

{information}

REASONING INSTRUCTIONS:

{reasoning}
"""

    result = analyze_vehicle(
        crew_input
    )

    return {
        "final_result": str(result)
    }


# ============================================
# BUILD LANGGRAPH
# ============================================

workflow = StateGraph(VehicleState)


workflow.add_node(
    "load_documents",
    load_documents
)

workflow.add_node(
    "retrieve_information",
    retrieve_information
)

workflow.add_node(
    "reasoning",
    reasoning_node
)

workflow.add_node(
    "crewai",
    crewai_node
)


# ============================================
# EDGES
# ============================================

workflow.add_edge(
    START,
    "load_documents"
)

workflow.add_edge(
    "load_documents",
    "retrieve_information"
)

workflow.add_edge(
    "retrieve_information",
    "reasoning"
)

workflow.add_edge(
    "reasoning",
    "crewai"
)

workflow.add_edge(
    "crewai",
    END
)


# ============================================
# COMPILE
# ============================================

app = workflow.compile()


# ============================================
# TEST
# ============================================

if __name__ == "__main__":

    print("\n===================================")
    print("       USED CAR AI")
    print("       LANGGRAPH WORKFLOW")
    print("===================================")

    initial_state = {
        "pdf_files": [],
        "vehicle_information": "",
        "reasoning": "",
        "final_result": ""
    }

    result = app.invoke(
        initial_state
    )

    print("\n===================================")
    print("       FINAL VEHICLE ASSESSMENT")
    print("===================================")

    print(result["final_result"])
