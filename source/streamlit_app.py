import streamlit as st
import umodel

# Show title and description.
st.title("💬 HYDAC-GPT")
st.write(
    "Welcome to HYDAC-GPT demo version."
)

# Create a session state variable to store the chat history. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("Type your question here."):

    # Store the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display the prompt
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a response
    ai_answer = umodel.chain.invoke(prompt)

    # Display assistant response in chat message container
    with st.chat_message("HYDAC"):
        st.markdown(ai_answer)

    st.session_state.messages.append({"role": "assistant", "content": ai_answer})
