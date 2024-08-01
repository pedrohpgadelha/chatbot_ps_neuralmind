import json
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Initializing the chatbot.
# Load and process documents.
file_path = "data/resolucao_vest_unicamp.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# Initialize the language model.
llm = ChatOpenAI(model="gpt-4o")

# Split documents into manageable text chunks.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create vector store for document retrieval.
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Setup system prompt for the chat.
system_prompt = (
    "Você é um chatbot inteligente para responder dúvidas acerca do vestibular"
    "da unicamp de 2025. Utilize os seguintes pedaços de texto para responder a"
    "pergunta. Se você não souber uma resposta, diga que não sabe e recomende"
    "ao usuário ler a resolução da comvest acerca do vestibular. Use três"
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

def load_test_data(file_path):
    '''
    Retorna os dados contidos no arquivo de perguntas e respostas para teste.
    '''
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Initialize your RAG chain.
def test_rag_chain(rag_chain, test_data):
    predictions = []
    references = []
    
    for item in test_data:
        input_question = item['question']
        ground_truth_answer = item['answer']
        results = rag_chain.invoke({"input": input_question})
        generated_answer = results['answer']
        
        predictions.append(generated_answer)
        references.append([ground_truth_answer])
    
    return predictions, references

def evaluate(predictions, references):
    # Flatten references for Rouge
    flat_references = [ref[0] for ref in references]
    
    # BLEU
    bleu = corpus_bleu(references, predictions)
    
    # ROUGE
    rouge = Rouge()
    scores = rouge.get_scores(predictions, flat_references, avg=True)
    
    return {
        "bleu": bleu,
        "rouge": scores
    }


# Run the tests.
# Load the test data.
test_data = load_test_data('data/questions_answers.json')

# Run the questions through the model and get the answers and expected answers.
predictions, references = test_rag_chain(rag_chain, test_data)

# Get and print evaluation results.
evaluation_results = evaluate(predictions, references)
print(evaluation_results)