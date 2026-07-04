from dotenv import load_dotenv
load_dotenv()  # This must run first to populate the environment variables

import streamlit as st
from tvm_agent import process_tvm_request

# Set up clean page configuration
st.set_page_config(
    page_title="AI Actuarial TVM Platform",
    page_icon="📈",
    layout="centered"
)

# ==============================================================================
# THEME POLISH
# Button/background colors live in .streamlit/config.toml. This block handles
# what config.toml can't: markdown table contrast inside chat bubbles, and
# breathing room on the reasoning text.
# ==============================================================================
st.markdown(
    """
    <style>
    div[data-testid="stMarkdownContainer"] table td,
    div[data-testid="stMarkdownContainer"] table th {
        color: #E8E8E8 !important;
    }
    div[data-testid="stMarkdownContainer"] table th {
        color: #1F9D6B !important;
        border-bottom: 1px solid #2A2A2A !important;
    }
    div[data-testid="stMarkdownContainer"] table td {
        border-bottom: 1px solid #232323 !important;
    }
    div[data-testid="stMarkdownContainer"] p {
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ASSISTANT_AVATAR = "🧮"
USER_AVATAR = "🙋"

WELCOME_MESSAGE = (
    "👋 Hi, I'm your AI actuarial assistant. Ask me a Time Value of Money "
    "question — loans, rate conversions, force of interest, or equation of "
    "value — and I'll walk through the reasoning, show the formula, and "
    "verify the math like an actuary would. Try one of the examples below, "
    "or just type your own question."
)

# ==============================================================================
# EXAMPLE QUESTIONS
# ==============================================================================
example_1 = "Calculate the monthly payment for a 30-year balanced loan of $300,000 at an annual nominal interest rate of 6% compounded monthly."
example_2 = "Convert a nominal interest rate of 8% compounded quarterly to an effective annual rate."
example_3 = "Find the accumulation factor from time t=0 to t=2 if the force of interest is variable and defined by delta(t) = 0.02 + 0.01 * t."
example_4 = "A borrower receives $5,000 today and repays $2,000 at the end of year 1 and $3,200 at the end of year 2. Is this loan fair at a 5% effective annual rate?"

# ==============================================================================
# CONVERSATION STATE
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]

# Header
st.title("📈 AI-Driven Time Value of Money Solver")
st.caption("A chat-based actuarial engine — built for students learning the concepts and professionals who want a fast, verified answer.")

# Reset button
_, reset_col = st.columns([4, 1])
with reset_col:
    if st.button("🔄 New chat", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
        st.rerun()

# Example prompt chips - collapsed once a real conversation is underway
with st.expander("💡 Try an example question", expanded=(len(st.session_state.messages) == 1)):
    ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
    clicked_prompt = None
    with ex_col1:
        if st.button("Loan", use_container_width=True):
            clicked_prompt = example_1
    with ex_col2:
        if st.button("Nominal → Effective", use_container_width=True):
            clicked_prompt = example_2
    with ex_col3:
        if st.button("Force of Interest", use_container_width=True):
            clicked_prompt = example_3
    with ex_col4:
        if st.button("Equation of Value", use_container_width=True):
            clicked_prompt = example_4

    if clicked_prompt:
        st.session_state.messages.append({"role": "user", "content": clicked_prompt})
        with st.spinner("Working through the math..."):
            response = process_tvm_request(clicked_prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ==============================================================================
# RENDER CONVERSATION HISTORY
# ==============================================================================
for msg in st.session_state.messages:
    avatar = ASSISTANT_AVATAR if msg["role"] == "assistant" else USER_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ==============================================================================
# CHAT INPUT
# ==============================================================================
if prompt := st.chat_input("Type your TVM question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Working through the math..."):
            response = process_tvm_request(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer info
st.markdown("---")
st.caption("Actuarial AI Platform Prototype | Powered by Gemini 2.5 Flash & Pure Mathematical Backend")