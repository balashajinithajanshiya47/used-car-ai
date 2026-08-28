import streamlit as st
from graph import car_graph

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="Used Car AI Assistant",
    page_icon="🚗",
    layout="centered"
)


# ============================================
# Title
# ============================================

st.title("🚗 Used Car AI Assistant")

st.write(
    "Ask questions about the vehicle's inspection, "
    "service history, and insurance history."
)


# ============================================
# User Question
# ============================================

question = st.text_input(
    "Ask a question about the vehicle:",
    placeholder="Example: Did this vehicle have any insurance claims?"
)


# ============================================
# Ask AI Button
# ============================================

if st.button("Ask AI"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching vehicle documents..."):

            documents = search_documents(question)

        if not documents:

            st.warning(
                "No relevant information was found "
                "in the vehicle documents."
            )

        else:

            with st.spinner("Generating AI answer..."):

                answer = generate_answer(
                    question,
                    documents
                )

            # --------------------------------
            # AI Answer
            # --------------------------------

            st.subheader("🤖 AI Answer")

            st.write(answer)


            # --------------------------------
            # Sources
            # --------------------------------

            st.subheader("📄 Sources")

            sources = set()

            for document in documents:

                source = document.metadata.get("source")

                if source:

                    sources.add(source)

            for source in sources:

                st.write(f"• {source}")