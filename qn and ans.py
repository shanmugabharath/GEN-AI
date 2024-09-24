mport os
import requests
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM

# 1. Setup Gemini Pro API
API_KEY = AIzaSyBzUi4AFJL-yDEGj7kLEOAc1Q_7bPZ72dA
API_ENDPOINT = 'https://api.geminivisionpro.com/v1/generate'

class GeminiProLLM(LLM):
    def _call(self, prompt: str, stop=None):
        headers = {'x-api-key': API_KEY}
        data = {'prompt': prompt}
        response = requests.post(API_ENDPOINT, json=data, headers=headers)
        response.raise_for_status()
        return response.json().get('output', '')

    def _identifying_params(self):
        return {}

# 2. Load Documents and Build the Vector Store
documents = [
    Document(page_content="What is Gemini Pro?", metadata={"source": "gemini_intro.txt"}),
    Document(page_content="Gemini Pro is an API for building AI applications.", metadata={"source": "gemini_intro.txt"}),
    # Add more documents as needed
]

# 3. Embed and store documents in FAISS
embedding_model = OpenAIEmbeddings()  # or use any other embedding method
vector_store = FAISS.from_documents(documents, embedding_model)

# 4. Setup the Retrieval Augmented Generation (RAG) Chain
retriever = vector_store.as_retriever()

# Define a custom prompt template
prompt_template = """
You are an AI expert with access to a set of documents. Based on the context below, generate an accurate response.
Context: {context}
Question: {question}
Answer:
"""

qa_chain = RetrievalQA.from_chain_type(
    llm=GeminiProLLM(),
    chain_type="stuff",  # Stuff means passing all the retrieved documents to the LLM
    retriever=retriever,
    prompt=PromptTemplate(input_variables=["context", "question"], template=prompt_template),
)

# 5. Function to Ask Questions
def ask_question(query):
    return qa_chain.run(query)

# 6. Run the App
if __name__ == "__main__":
    question = "What is Gemini Pro?"
    answer = ask_question(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
