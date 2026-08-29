from typing import TypedDict
import os

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

from faiss_db import create_faiss_database, search_faiss


class VehicleState(TypedDict):
    pdf_files: list
    vehicle_information: str
    reasoning: str
    final_result: str


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not configured.")


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0
)


def load_documents(state: VehicleState):

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


def retrieve_information(state: VehicleState):

    pdf_files = state["pdf_files"]

    if not pdf_files:
        return {
            "vehicle_information":
                "No vehicle PDF documents were found."
        }

    index = create_faiss_database(pdf_files)

    insurance = search_faiss(
        index,
        "insurance claim accident damage insurance history",
        top_k=3
    )

    service = search_faiss(
        index,
        "service maintenance repair servicing brake transmission engine sensor",
        top_k=3
    )

    inspection = search_faiss(
        index,
        "inspection engine transmission brakes tyres body warning light mechanical condition",
        top_k=3
    )

    vehicle_information = ""

    vehicle_information += "\nINSURANCE HISTORY:\n"
    for result in insurance:
        vehicle_information += result["text"] + "\n"

    vehicle_information += "\nSERVICE HISTORY:\n"
    for result in service:
        vehicle_information += result["text"] + "\n"

    vehicle_information += "\nINSPECTION REPORT:\n"
    for result in inspection:
        vehicle_information += result["text"] + "\n"

    return {
        "vehicle_information": vehicle_information
    }


def reasoning_node(state: VehicleState):

    information = state["vehicle_information"]

    prompt = f"""
You are a used-car reasoning assistant.

Analyze the vehicle information below.

Identify:

1. Important insurance events
2. Important service events
3. Inspection concerns
4. Positive points
5. Items that should be checked before purchase

Rules:

- Use ONLY the supplied information.
- Do not invent facts.
- Do not assume missing information.
- Clearly separate facts from recommendations.

VEHICLE INFORMATION:

{information}
"""

    response = llm.invoke(prompt)

    return {
        "reasoning": response.content
    }


def final_assessment_node(state: VehicleState):

    information = state["vehicle_information"]
    reasoning = state["reasoning"]

    prompt = f"""
You are a senior used-car advisor.

Create a final assessment using ONLY the vehicle
information provided below.

Do not invent facts.

Clearly distinguish facts from recommendations.

Use this structure:

1. OVERALL ASSESSMENT
2. INSURANCE HISTORY
3. SERVICE HISTORY
4. INSPECTION CONDITION
5. POSITIVE POINTS
6. POTENTIAL CONCERNS
7. RECOMMENDED CHECKS BEFORE PURCHASE
8. FINAL VERDICT

VEHICLE INFORMATION:

{information}

REASONING:

{reasoning}
"""

    response = llm.invoke(prompt)

    return {
        "final_result": response.content
    }


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
    "final_assessment",
    final_assessment_node
)

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
    "final_assessment"
)

workflow.add_edge(
    "final_assessment",
    END
)

app = workflow.compile()


if __name__ == "__main__":

    initial_state = {
        "pdf_files": [],
        "vehicle_information": "",
        "reasoning": "",
        "final_result": ""
    }

    result = app.invoke(initial_state)

    print("\n===================================")
    print("FINAL VEHICLE ASSESSMENT")
    print("===================================")

    print(result["final_result"])