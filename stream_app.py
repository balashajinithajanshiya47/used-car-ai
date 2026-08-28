import streamlit as st
import tempfile
import os

from langgraph_flow import app


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Used Car AI",
    page_icon="🚗",
    layout="wide"
)


# ============================================
# TITLE
# ============================================

st.title("🚗 Used Car AI")

st.write(
    "AI-powered used vehicle analysis using "
    "FAISS, LangGraph, CrewAI and Ollama."
)


st.divider()


# ============================================
# SIDEBAR
# ============================================

st.sidebar.header("Vehicle Documents")

st.sidebar.write(
    "Upload the vehicle documents below."
)


# ============================================
# PDF UPLOAD
# ============================================

insurance_file = st.sidebar.file_uploader(
    "Insurance History PDF",
    type=["pdf"]
)

service_file = st.sidebar.file_uploader(
    "Service History PDF",
    type=["pdf"]
)

inspection_file = st.sidebar.file_uploader(
    "Vehicle Inspection PDF",
    type=["pdf"]
)


st.divider()


# ============================================
# ANALYZE BUTTON
# ============================================

if st.button(
    "🔍 Analyze Vehicle",
    type="primary"
):

    if not insurance_file:
        st.error(
            "Please upload the Insurance History PDF."
        )

    elif not service_file:
        st.error(
            "Please upload the Service History PDF."
        )

    elif not inspection_file:
        st.error(
            "Please upload the Vehicle Inspection PDF."
        )

    else:

        with st.spinner(
            "Analyzing vehicle... Please wait."
        ):

            try:

                # --------------------------------
                # Create temporary directory
                # --------------------------------

                with tempfile.TemporaryDirectory() as temp_dir:

                    insurance_path = os.path.join(
                        temp_dir,
                        "insurance_history.pdf"
                    )

                    service_path = os.path.join(
                        temp_dir,
                        "service_history.pdf"
                    )

                    inspection_path = os.path.join(
                        temp_dir,
                        "vehicle_inspection.pdf"
                    )


                    # --------------------------------
                    # Save uploaded files
                    # --------------------------------

                    with open(
                        insurance_path,
                        "wb"
                    ) as f:

                        f.write(
                            insurance_file.getbuffer()
                        )


                    with open(
                        service_path,
                        "wb"
                    ) as f:

                        f.write(
                            service_file.getbuffer()
                        )


                    with open(
                        inspection_path,
                        "wb"
                    ) as f:

                        f.write(
                            inspection_file.getbuffer()
                        )


                    # --------------------------------
                    # Run LangGraph
                    # --------------------------------

                    initial_state = {

                        "pdf_files": [
                            insurance_path,
                            service_path,
                            inspection_path
                        ],

                        "vehicle_information": "",

                        "reasoning": "",

                        "final_result": ""
                    }


                    result = app.invoke(
                        initial_state
                    )


                    # --------------------------------
                    # Display Result
                    # --------------------------------

                    st.success(
                        "Vehicle analysis completed!"
                    )


                    st.subheader(
                        "📊 Final Vehicle Assessment"
                    )


                    st.markdown(
                        result["final_result"]
                    )


            except Exception as e:

                st.error(
                    "An error occurred while analyzing "
                    "the vehicle."
                )

                st.exception(e)


# ============================================
# INFORMATION SECTION
# ============================================

st.divider()

st.subheader("🔧 Technology Stack")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("📚 FAISS")
    st.caption("Vector Database")

with col2:
    st.write("🔄 LangGraph")
    st.caption("Workflow")

with col3:
    st.write("🤖 CrewAI")
    st.caption("Multi-Agent AI")

with col4:
    st.write("🧠 Ollama")
    st.caption("llama3.2:latest")

