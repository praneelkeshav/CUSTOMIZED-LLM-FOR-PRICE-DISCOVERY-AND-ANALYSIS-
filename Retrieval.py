from langchain_google_genai import GoogleGenerativeAI
import getpass
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import vertexai
from vertexai.generative_models import GenerativeModel
import google.generativeai as genai
#import langchain
#langchain.config.allow_dangerous_deserialization = True


google_api_key= 'AIzaSyBIvvpkTdcBM8YpfwpgGS_LwiHSsawc3iY'
llm=GoogleGenerativeAI(google_api_key=google_api_key,model="models/text-bison-001", temperature=0.9)

google_api_key = 'AIzaSyBIvvpkTdcBM8YpfwpgGS_LwiHSsawc3iY'
query_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query", google_api_key=google_api_key)
doc_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document", google_api_key=google_api_key)
vectordb_file_path="vector_index"

 
def create_vector_db():
    loader = CSVLoader(file_path='D:/INTERN/2nd Year/Project/Codes/main/Sample_Data_Description.csv', source_column="Material_Description", encoding="UTF-8")

    # Store the loaded data in the 'data' variable
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=query_embeddings)
    vectordb.save_local(vectordb_file_path)


def get_qa_chain(allow_dangerous_deserialization=True):

    # Create a FAISS instance for vector database from 'data'
    #vectordb = FAISS.load_local(vectordb_file_path,query_embeddings)
    #Create a retriever for querying the vector database
    
    new_db = FAISS.load_local("vector_index",query_embeddings,allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(score_threshold = 0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    If the answer is not found in the context, try to make up an answer from searching the web.If the question is not related to the context,provide 
    the relavant answer to the question using the llm function above used,try to chat with them and be a bit creative,provide image if needed,take the image from  the web.
    Show the details from the dataset if the question is related to the dataset.give the result in two parts,one as the answer from web and another one from the dataset.


    

    CONTEXT: {context}

    QUESTION: {question}"""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    from langchain.chains import RetrievalQA

    chain = RetrievalQA.from_chain_type(llm=llm,
                                chain_type="stuff",
                                retriever=retriever,
                                input_key="query",
                                return_source_documents=True,
                                chain_type_kwargs=chain_type_kwargs)
    if allow_dangerous_deserialization:
        pass
    else:
        
        pass
    return chain


if __name__=="__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Price and description about Rope suspended platform"))



