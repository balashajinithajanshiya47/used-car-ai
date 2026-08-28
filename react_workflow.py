
from langchain_ollama import OllamaLLM

from faiss_db import (
    create_faiss_database,
    search_faiss
)

from crew import analyze_vehicle

import os


# ============================================
# Ollama Configuration
# ============================================

llm = OllamaLLM(
    model="llama3.2:latest"
)


# ============================================
# Create FAISS Database
# ============================================

def load_vehicle_documents():

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

        print("No PDF documents found.")

        return None

    print("\n===================================")
    print("       LOADING VEHICLE DOCUMENTS")
    print("===================================")

    index = create_faiss_database(
        pdf_files
    )

    return index


# ============================================
# ReAct-Style Reasoning
# ============================================

def react_reasoning(vehicle_information):

    prompt = f"""
You are a used-car reasoning assistant.

Analyze the following vehicle information.

VEHICLE INFORMATION:

{vehicle_information}

Think through the information step by step.

Check:

1. Insurance history
2. Service history
3. Inspection condition
4. Positive points
5. Potential concerns
6. Recommended checks

Do not invent information.

Give a concise reasoning summary that can be
passed to a senior used-car advisor.
"""

    response = llm.invoke(prompt)

    return response


# ============================================
# Main Workflow
# ============================================

def run_workflow():

    print("\n===================================")
    print("       USED CAR REACT WORKFLOW")
    print("===================================")


    # ----------------------------------------
    # Step 1: Load FAISS
    # ----------------------------------------

    index = load_vehicle_documents()

    if index is None:

        return


    # ----------------------------------------
    # Step 2: Retrieve Insurance Information
    # ----------------------------------------

    insurance_results = search_faiss(
        index,
        "What is the insurance claim history?",
        top_k=3
    )


    # ----------------------------------------
    # Step 3: Retrieve Service Information
    # ----------------------------------------

    service_results = search_faiss(
        index,
        "What is the vehicle service and maintenance history?",
        top_k=3
    )


    # ----------------------------------------
    # Step 4: Retrieve Inspection Information
    # ----------------------------------------

    inspection_results = search_faiss(
        index,
        "What are the vehicle inspection and mechanical concerns?",
        top_k=3
    )


    # ----------------------------------------
    # Step 5: Combine Results
    # ----------------------------------------

    vehicle_information = ""

    vehicle_information += "\nINSURANCE INFORMATION:\n"

    for result in insurance_results:

        vehicle_information += (
            result["text"] + "\n"
        )


    vehicle_information += "\nSERVICE INFORMATION:\n"

    for result in service_results:

        vehicle_information += (
            result["text"] + "\n"
        )


    vehicle_information += "\nINSPECTION INFORMATION:\n"

    for result in inspection_results:

        vehicle_information += (
            result["text"] + "\n"
        )


    # ----------------------------------------
    # Step 6: ReAct Reasoning
    # ----------------------------------------

    print("\n===================================")
    print("       REACT REASONING")
    print("===================================")

    reasoning = react_reasoning(
        vehicle_information
    )

    print("\nReasoning Summary:")
    print(reasoning)


    # ----------------------------------------
    # Step 7: Send to CrewAI
    # ----------------------------------------

    print("\n===================================")
    print("       STARTING CREWAI")
    print("===================================")

    final_information = f"""
VEHICLE INFORMATION:

{vehicle_information}

REASONING SUMMARY:

{reasoning}
"""

    final_result = analyze_vehicle(
        final_information
    )


    # ----------------------------------------
    # Step 8: Final Result
    # ----------------------------------------

    print("\n===================================")
    print("       FINAL VEHICLE ASSESSMENT")
    print("===================================")

    print(final_result)


# ============================================
# Run
# ============================================

if __name__ == "__main__":

    run_workflow()
