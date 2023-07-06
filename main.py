from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from configs import OPENAI_API_KEY
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *
from PIL import Image
import streamlit as st

# def initialize_session_state():
#     if "buffer_memory" not in st.session_state:
#         st.session_state.buffer_memory = None

# # Call the initialization function at the start of your Streamlit app
# initialize_session_state()
# # Access the buffer_memory attribute
#buffer_memory = st.session_state.buffer_memory

image = Image.open('image/company.PNG')
st.image(image, caption='Company Data')
st.subheader("Claim Liability Chatbot")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Good Day! How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0.2)

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=5,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""You are a claim assistant bot that will answer questions from a claims adjuster. You will counsel them to follow company guidelines and help them make better decisions. 
        Your responses should be informational, logical, and provide great context. Answer the question as truthfully as possible using the provided context. Your reasoning should be rigorous, intelligent, and defensible. Your responses should be safe, free of harm, and non-controversial. '''
    """)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            #print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
