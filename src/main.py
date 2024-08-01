import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load and process documents.
file_path = "data/resolucao_vest_unicamp.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# Split documents into text chunks..
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Initialize the language model.
llm = ChatOpenAI(model="gpt-4o")

# Create vector store for data retrieval.
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Setup system prompt for the chat.
system_prompt = (
    "Você é um chatbot inteligente para responder dúvidas acerca do vestibular"
    "da unicamp de 2025. Utilize os seguintes pedaços de texto para responder a"
    "pergunta. Se você não souber uma resposta, diga que não sabe. Use três"
    "frases no máximo e mantenha suas respostas concisas.\n\n{context}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Creating chains for question answering.
question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Streamlit app setup.
st.title('Chatbot de Dúvidas do Vestibular Unicamp 2025')

# User input for query.
user_query = st.text_input("Escreva sua dúvida sobre o Vestibular Unicamp 2025")

if st.button('Gerar resposta'):
    # Query the chain.
    results = rag_chain.invoke({"input": user_query})
    
    # Display the results.
    st.write("Answer:", results['answer'])