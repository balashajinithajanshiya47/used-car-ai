import os

from crewai import Agent, Task, Crew, Process, LLM


# ============================================
# GROQ CONFIGURATION
# ============================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set."
    )


# ============================================
# GROQ LLM
# ============================================

llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=GROQ_API_KEY
)


# ============================================
# INSURANCE AGENT
# ============================================

insurance_agent = Agent(
    role="Insurance History Analyst",

    goal=(
        "Analyze the vehicle insurance history and "
        "identify insurance claims, accident history, "
        "damage history, and important insurance concerns."
    ),

    backstory=(
        "You are an experienced used-car insurance "
        "history analyst. You carefully examine "
        "vehicle records and never invent information."
    ),

    llm=llm,

    verbose=False,

    allow_delegation=False
)


# ============================================
# SERVICE AGENT
# ============================================

service_agent = Agent(
    role="Vehicle Service History Analyst",

    goal=(
        "Analyze the vehicle service history and "
        "identify maintenance, repairs, replacements, "
        "and important service events."
    ),

    backstory=(
        "You are an experienced automotive service "
        "history analyst. You identify important "
        "maintenance events from vehicle records."
    ),

    llm=llm,

    verbose=False,

    allow_delegation=False
)


# ============================================
# INSPECTION AGENT
# ============================================

inspection_agent = Agent(
    role="Vehicle Inspection Analyst",

    goal=(
        "Analyze the vehicle inspection report and "
        "identify mechanical, tyre, brake, engine, "
        "transmission, and body-condition concerns."
    ),

    backstory=(
        "You are an experienced vehicle inspection "
        "specialist. You assess inspection information "
        "carefully and identify potential concerns."
    ),

    llm=llm,

    verbose=False,

    allow_delegation=False
)


# ============================================
# FINAL VEHICLE ADVISOR
# ============================================

final_agent = Agent(
    role="Senior Used Car Advisor",

    goal=(
        "Combine the insurance, service, and inspection "
        "analysis and provide a clear overall assessment "
        "of the vehicle."
    ),

    backstory=(
        "You are a senior used-car advisor. You review "
        "multiple vehicle reports and provide balanced, "
        "evidence-based recommendations."
    ),

    llm=llm,

    verbose=False,

    allow_delegation=False
)


# ============================================
# ANALYZE VEHICLE
# ============================================

def analyze_vehicle(vehicle_information):

    # ----------------------------------------
    # Insurance
    # ----------------------------------------

    insurance_task = Task(
        description=f"""
Analyze the following vehicle information.

Focus specifically on:

- Insurance claims
- Accident-related information
- Damage
- Claim dates
- Important insurance concerns

Vehicle information:

{vehicle_information}

Only use the information provided.

Do not invent facts.
""",

        expected_output=(
            "A concise summary of the vehicle's insurance "
            "history and important concerns."
        ),

        agent=insurance_agent
    )


    # ----------------------------------------
    # Service
    # ----------------------------------------

    service_task = Task(
        description=f"""
Analyze the following vehicle information.

Focus specifically on:

- Regular servicing
- Repairs
- Replaced components
- Maintenance dates
- Mileage at service
- Important maintenance concerns

Vehicle information:

{vehicle_information}

Only use the information provided.

Do not invent facts.
""",

        expected_output=(
            "A concise summary of the vehicle's service "
            "history and maintenance concerns."
        ),

        agent=service_agent
    )


    # ----------------------------------------
    # Inspection
    # ----------------------------------------

    inspection_task = Task(
        description=f"""
Analyze the following vehicle information.

Focus specifically on:

- Engine
- Transmission
- Brakes
- Tyres
- Body
- Warning lights
- Inspection recommendations
- Potential mechanical concerns

Vehicle information:

{vehicle_information}

Only use the information provided.

Do not invent facts.
""",

        expected_output=(
            "A concise summary of the vehicle inspection "
            "and important mechanical concerns."
        ),

        agent=inspection_agent
    )


    # ----------------------------------------
    # Final Assessment
    # ----------------------------------------

    final_task = Task(
        description=f"""
Review the vehicle information and the analysis
performed by the other specialists.

Create a final used-car assessment.

Include:

1. Insurance history
2. Service history
3. Inspection condition
4. Positive points
5. Potential concerns
6. Recommended checks before purchase

Only use information provided in the vehicle records.

Do not invent facts.

Clearly distinguish facts from recommendations.

Vehicle information:

{vehicle_information}
""",

        expected_output=(
            "A structured overall used-car assessment "
            "based only on the provided vehicle information."
        ),

        agent=final_agent,

        context=[
            insurance_task,
            service_task,
            inspection_task
        ]
    )


    # ----------------------------------------
    # Crew
    # ----------------------------------------

    crew = Crew(
        agents=[
            insurance_agent,
            service_agent,
            inspection_agent,
            final_agent
        ],

        tasks=[
            insurance_task,
            service_task,
            inspection_task,
            final_task
        ],

        process=Process.sequential,

        verbose=False
    )


    # ----------------------------------------
    # Run
    # ----------------------------------------

    result = crew.kickoff()

    return result


# ============================================
# LOCAL TEST
# ============================================

if __name__ == "__main__":

    print("Testing CrewAI...")

    sample_information = """
2019 Example Sedan

Insurance:
One rear bumper insurance claim in 2021.

Service:
Regular servicing at 15,000 km and 30,000 km.
Brake pads replaced at 45,000 km.

Inspection:
Engine starts normally.
No visible oil leakage.
Front tyres approximately 30% remaining.
Professional diagnostic scan recommended.
"""

    result = analyze_vehicle(
        sample_information
    )

    print("\n===================================")
    print("FINAL RESULT")
    print("===================================")

    print(result)