import streamlit as st
import enterprise_model
import light_model
from light_model import pc_client

# Show title and description.
st.title("💬 HYDAC-GPT")
st.write(
    "Welcome to HYDAC-GPT! Ask any technical question!"
)
selected_model = st.selectbox(label='Select your preferred model.', index=0, placeholder='Light or Enterprise',
                              options=('Light', 'Enterprise'))

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

    if selected_model == 'Light':
        # Generate a response
        ai_answer = light_model.chain.invoke({'context': "\n\n".join(pc_client.vector_search(query=prompt)),
                                              'q': prompt}).content
    elif selected_model == 'Enterprise':
        ai_answer = enterprise_model.chain.invoke(prompt)
    else:
        ai_answer = "Error!"

    # Display assistant response in chat message container
    with st.chat_message("HYDAC-GPT"):
        st.markdown(selected_model)
        st.markdown(ai_answer)

    st.session_state.messages.append({"role": "HYDAC-GPT", "content": ai_answer})
