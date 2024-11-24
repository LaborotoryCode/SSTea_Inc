from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer
from time import time

from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS

#History of convo
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage

import pdfplumber
from langchain.document_loaders import TextLoader
from langchain.schema import Document

model_id = "meta-llama/Llama-3.2-3B-Instruct"

import os

file_path = r"C:\Users\ayaan\Desktop\.venv\1706.03762.pdf"
if not os.path.exists(file_path):
    print(f"File does not exist: {file_path}")
else:
    print(f"File found: {file_path}")

device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

#Initialise model
time_start = time()
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    max_new_tokens = 2024
)  

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
time_end = time()

#Query Pipeline
time_start = time()
query_pipeline = transformers.pipeline(
    "text-generation",
    model = model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens = 1024
)
time_end = time()

print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024**3} GB")
"""
#Testing function
def test_model(tokenizer, pipeline, prompt_to_test):
    time_start = time()
    sequences = pipeline(
        prompt_to_test,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    time_end = time()
    print(f"Test inference: {round(time_end-time_start, 3)} sec.")
    i = 1
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

#Test our pipeline
#print(test_model(tokenizer,
#           query_pipeline,
#           "Explain The state of the union in 100 words"))
"""
llm = HuggingFacePipeline(pipeline=query_pipeline)
print(llm(prompt=""))

#Load documents

document_path = file_path
document_text = ""

with pdfplumber.open(document_path) as pdf:
    for page in pdf.pages:
        document_text += page.extract_text()

document = Document(page_content=document_text)
documents = [document]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

#Creating vector embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device" : "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

faiss_vectordb = FAISS.from_documents(documents=all_splits,
                                 embedding=embeddings)

retriever = faiss_vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
)

#Test Retrieval
def test_rag(qa, query):
    print(f"Query: {query}\n")
    time_start = time()
    result = qa(query)
    time_end = time()
    print(f"Inference time: {round(time_end-time_start, 3)} sec.")
    print("\nResult: ", result)

query = """Ask me a question from the document and provide the answer too"""

print(test_rag(qa, query))

"""
#Keep conversational history
query = input("Please enter a query for the documents: ")

workflow = StateGraph(state_schema=MessagesState)

def call_rag(state: MessagesState):
    system_prompt = (
        "You must answer all questions to the best of you ability based on the documents given."
        "If you do not know the answer do not guess"
        "Pretend that you are my professor"
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    print("Answering")
    response = qa(messages)
    answer = {"messages": response}
    index = str(answer)
    index = index.find("Helpful Answer: ")
    print("index:", index)
    return answer[index::]
   

workflow.add_node("model", call_rag)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

while True:
    query = input("Please enter a query for the documents: ")

    app.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": "1"}}
    )
"""
