import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
import streamlit as st
import openai
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
import chardet


load_dotenv()
os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.environ['OPENAI_API_KEY']


chat = ChatOpenAI(
    model="gpt-4o",
    temperature=0.4,
)

with open(r'cleaned_data.txt', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

#with open(r'C:\Users\Bolin\Desktop\dataset\econ.txt', 'rb') as f:
#    result = chardet.detect(f.read())
#    encoding = result['encoding']

# Load documents
loader1 = TextLoader(r'cleaned_data.txt', encoding=encoding)
pages1 = loader1.load_and_split()

loader2 = TextLoader(r'mental_health_support.txt', encoding='utf-8')
pages2 = loader2.load_and_split()

loader3 = TextLoader("econ_example.txt", encoding="utf-8")
pages3 = loader3.load_and_split()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
docs1 = text_splitter.split_documents(pages1)
docs2 = text_splitter.split_documents(pages2)
docs3 = text_splitter.split_documents(pages3)

# Create embeddings
embed_model = OpenAIEmbeddings()

# Create vector stores and retrievers
vectorstore1 = Chroma.from_documents(documents=docs1, embedding=embed_model, collection_name="cleaned_data_docs")
vectorstore2 = Chroma.from_documents(documents=docs2, embedding=embed_model, collection_name="mental_health_docs")
vectorstore3 = Chroma.from_documents(documents=docs3, embedding=embed_model, collection_name="econ_docs")

retriever1 = vectorstore1.as_retriever(k=2)
retriever2 = vectorstore2.as_retriever(k=2)
retriever3 = vectorstore3.as_retriever(k=2)


# Create ChatPromptTemplate
chat_history = ChatMessageHistory()

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
         You are HopeBot, a professional psychotherapist specialising in Cognitive Behavioural Therapy (CBT). Your role is to focus on your clients' words and emotions, guiding them to reflect on their thoughts and behaviours through open-ended questions and guiding them through the PHQ-9 test. Always show empathy and understanding of their feelings and help them to recognise how their behaviour affects their emotions.Your responses should not be too long or presented in bullet point form, which is too mechanical, and all your responses should be spoken. If a customer comes to you for advice, give two or three at a time.
You need to complete three tasks in turn:
Task 1: As a professional counsellor, you should begin by greeting the client warmly and start a casual conversation asking about their current situation. Do not exceed 20 rounds of dialogue in this task and transition to introducing the PHQ-9 when appropriate, if the user states twice or more that they have nothing to share or when the dialogue up to 20 rounds, you must ask the user if they would like to take the PHQ-9 test and give a brief introduction to the PHQ-9, communicating that this can be seen as a tool to help understand their feelings and offer support.

Task 2: After the user agrees to use the PHQ-9, ask each question in turn. Accurately categorise the user's answers as options A, B, C or D. If the user's answer is not precise enough, ambiguous or cannot be accurately categorised, ask the user to provide a clearer answer to ensure that the most accurate answer is collected. If the user answers A, they get 0 points; B, 1 point; C, 2 points; and D, 3 points. Track the score cumulatively without displaying it, and move to Task 3 after completing the test.

Task 3: You must tell the user of the total score in number on the PHQ-9. Show your calculation process. In the format: You scored X points. Here’s how each answer was interpreted: Question 1: X (X point), etc. And provide the appropriate depression severity results. Provide appropriate advice based on the results. If the depression is severe, give your advice and also encourage the user to seek professional help and provide them with a UK telephone helpline or email address (no more than 2 contacts). Be sure to make it clear that you are a virtual mental health assistant, not a doctor, and that whilst you will offer help, you are not a substitute for professional medical advice.
At the end you will need to provide a brief summary of your conversation, including the confusion raised by the user in Task 1, as well as their PHQ-9 test results, and your corresponding recommendations. You need to ask the user if they have any further questions about the result and answer them.

Please maintain the demeanour of a professional psychologist at all times and show empathy in your interactions. Please keep your responses concise and avoid giving long, repetitive answers.
Here is some additional background information to help guide your responses:\n\n{context}
        """),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Create the LLM chain with the language model and the prompt
document_chain = LLMChain(llm=chat, prompt=question_answering_prompt)

# Function to process input and return the chatbot's response
def get_assistant_response(messages):
    # Extract the user's last message (the latest user input)
    user_input = messages[-1]["content"]

    # Simulate chat history
    chat_history = ChatMessageHistory()
    for message in messages:
        chat_history.add_message(HumanMessage(content=message["content"]) if message["role"] == "user" else AIMessage(content=message["content"]))

    # Retrieve documents based on user input
    retriever_context = user_input  # Use user input as the query for document retrieval
    retrieved_docs1 = retriever1.get_relevant_documents(retriever_context)
    retrieved_docs2 = retriever2.get_relevant_documents(retriever_context)
    retrieved_docs3 = retriever3.get_relevant_documents(retriever_context)

    # Combine retrieved content into one context
    combined_context = "\n".join([doc.page_content for doc in retrieved_docs1 + retrieved_docs2])

    # Generate chatbot response with retrieved context
    response = document_chain.run(
        {
            "context": combined_context,  # Documents retrieved from retrievers
            "messages": chat_history.messages  # Conversation history
        }
    )

    # Return the assistant's response
    return response


def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file
        )
    return transcript

def text_to_speech(input_text):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=input_text
    )
    webm_file_path = "temp_audio_play.mp3"
    with open(webm_file_path, "wb") as f:
        response.stream_to_file(webm_file_path)
    return webm_file_path

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)