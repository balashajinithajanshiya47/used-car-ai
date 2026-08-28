

import os

from crewai import LLM

from faiss_db import (
    create_faiss_database,
    search_faiss
)

from crew import analyze_vehicle


# ============================================
# GROQ CONFIGURATION
# ============================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. "
        "Run: $env:GROQ_API_KEY='YOUR_NEW_KEY'"
    )

print("\nUsing Groq Cloud LLM...")

llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=GROQ_API_KEY
)


# ============================================
# CREATE FAISS DATABASE
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
# REACT-STYLE REASONING
# ============================================

def react_reasoning(vehicle_information):

    prompt = f"""
You are a used-car reasoning assistant.

Analyze the following vehicle information.

VEHICLE INFORMATION:

{vehicle_information}

Think through the information carefully.

Check:

1. Insurance history
2. Service history
3. Inspection condition
4. Positive points
5. Potential concerns
6. Recommended checks before purchase

IMPORTANT RULES:

- Use ONLY the information provided.
- Do not invent facts.
- Do not assume missing information.
- Clearly state when information is unavailable.
- Separate facts from recommendations.
- Keep the reasoning concise and useful.

Provide a reasoning summary that can be
passed to a senior used-car advisor.
"""

    response = llm.call(prompt)

    return response


# ============================================
# MAIN WORKFLOW
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

    print("\nSearching insurance information...")

    insurance_results = search_faiss(
        index,
        "What is the insurance claim history?",
        top_k=3
    )

    # ----------------------------------------
    # Step 3: Retrieve Service Information
    # ----------------------------------------

    print("\nSearching service information...")

    service_results = search_faiss(
        index,
        "What is the vehicle service and maintenance history?",
        top_k=3
    )

    # ----------------------------------------
    # Step 4: Retrieve Inspection Information
    # ----------------------------------------

    print("\nSearching inspection information...")

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
# RUN WORKFLOW
# ============================================

if __name__ == "__main__":

    run_workflow()
