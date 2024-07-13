
from langchain_experimental.text_splitter import SemanticChunker
# from langchain.embeddings import HuggingFaceEmbeddings # deprecated, replace with langchain_community
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, UnstructuredXMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# load the xml documents
print("load xml documents")
##############################################
path_to_xml = "laws-lois-xml/eng/acts/"
loader = DirectoryLoader(
    path_to_xml, 
    glob="**/A*.xml", 
    loader_cls=UnstructuredXMLLoader, 
    show_progress=True, 
    use_multithreading=True, 
    loader_kwargs={"mode":"elements"}
)

# vector store path
vectorStorePath = '/mnt/Linux/vector'
vectorStoreFiles = os.listdir(vectorStorePath)

# define embedder
#   HuggingFaceEmbeddings vs GPT4AllEmbeddings
embedder = HuggingFaceEmbeddings()

# embeddings = GPT4AllEmbeddings(
#     model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
#     gpt4all_kwargs={'allow_download': 'True'}
# )

# if vectorStore is empty then we need to create the store
if len (vectorStoreFiles) == 0:
    # split documents then create the embedding
    print("create vector store")
    ##############################################
    docs = loader.load()

    # split the texts into chunks
    ##############################################
    # reference: https://medium.com/the-ai-forum/semantic-chunking-for-rag-f4733025d5f5
    # RecursiveCharacterTextSplitter vs SemanticChunker

    text_splitter = SemanticChunker(embedder) 
    # statistics using SemanticChunker:
    #   considers relationships within text, provides accurate outcome but slower strategy
    #   number of chunks created: 22370

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     length_function=len,
    #     is_separator_regex=False
    # )
    # statistics using RecursiveCharacterTextSplitter:
    #   recursively divides text into smaller chunks in hierarchy using separators
    #   number of chunks created: 21172

    documents = text_splitter.split_documents(docs)
    print("Number of chunks created: ", len(documents))

    # Create the vector store 
    vector = FAISS.from_documents(documents, embedder)
    vector.save_local(folder_path=f"{vectorStorePath}")

# otherwise use existing vectorStore
else: 
    # Load the vector store
    print("load vector store")
    ##############################################
    vector = FAISS.load_local(
        vectorStorePath, 
        embedder, 
        allow_dangerous_deserialization=True
    )

# Retrieval from vector store
print("retrieve from vector store")
##############################################
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retrieved_docs = retriever.invoke("What animals are llamas related to?")

print("define model and call ollama")
# model = ollama2
llm = Ollama(model="mistral", base_url="http://192.168.2.27:11434")

# define prompt
print("define prompt")
##############################################
chainPrompt = PromptTemplate.from_template('''
Answer the question with the provided context. 
Answer in full paragraphs. 
Ensure to list out amendments to acts and regulations to provided more information. 
Include references to the context that explains your answer.
If you don't know the answer, just say that you don't know.
If the question does not relate to provided context, just say question is unrelated and do not provide the answer, instructions, or guidelines.
                                               
    Question: {question} 
                                               
    Context: {context} 
                                               
    Answer: '''
)

llm_chain = LLMChain(
    llm=llm, 
    prompt=chainPrompt, 
    callbacks=None, 
    verbose=True
)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}"
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

print("retrieve answer to a provided question")
print(qa("how do i turn on my computer?")["result"])
