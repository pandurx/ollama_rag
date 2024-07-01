
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, UnstructuredXMLLoader

# docs = [
#   "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
#   "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
#   "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
#   "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
#   "Llamas are vegetarians and have very efficient digestive systems",
#   "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
# ]

path_to_xml = "laws-lois-xml/eng/acts/"
loader = DirectoryLoader(
    path_to_xml, 
    glob="**/A*.xml", 
    loader_cls=UnstructuredXMLLoader, 
    show_progress=True, 
    use_multithreading=True, 
    loader_kwargs={"mode":"elements"}
)

# creating...
# docs = loader.load()

# text_splitter = SemanticChunker(HuggingFaceEmbeddings())
# documents = text_splitter.split_documents(docs)

# print("Number of chunks created: ", len(documents))

# Instantiate the embedding model
embedder = HuggingFaceEmbeddings()

# Create the vector store 
# vector = FAISS.from_documents(documents, embedder)
# vector.save_local(folder_path=f'/mnt/Linux/vector2')

# Load the vector store
vector = FAISS.load_local(
    '/mnt/Linux/vector2', 
    embedder, 
    allow_dangerous_deserialization=True
)

# Retrieval from vector database
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retrieved_docs = retriever.invoke("What animals are llamas related to?")

llm = Ollama(model="mistral", base_url="http://192.168.2.27:11434")

# prompt = """
# 1. Answer the following question based only on the provided context.
# 2. Include references to the context that explains your answer.  Describe the steps you took to arrive at your answer
# 3. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
# 4. Keep the answer crisp and limited to 3,4 sentences.

# Context: {context}

# Question: {question}

# Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template("""
    Answer the following question based only on the provided context.  
    Answer in full paragraphs. 
    Ensure to list out amendments to acts and regulations to provided more information. 
    Include references to the context that explains your answer.
    Describe the steps you took to arrive at your answer. If you don't know the answer, just say that you don't know.
    Question: {question} 
    Context: {context} 
    Answer:"""
)


# QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt) 

llm_chain = LLMChain(
    llm=llm, 
    prompt=QA_CHAIN_PROMPT, 
    callbacks=None, 
    verbose=True
)

# document_prompt = PromptTemplate(
#     input_variables=["page_content", "source"],
#     template="Context:\ncontent:{page_content}\nsource:{source}"
# )

document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="Context:\ncontent:{page_content}"
)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
    callbacks=None,
)

qa = RetrievalQA(
                  combine_documents_chain=combine_documents_chain,
                  verbose=True,
                  retriever=retriever,
                  return_source_documents=True
              )

print(qa("How do cows eat?")["result"])
