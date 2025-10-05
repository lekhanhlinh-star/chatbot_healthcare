import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

local_embeddings = OllamaEmbeddings(model="bge-m3")

model = ChatOllama(
    model="qwen2.5:14b",
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    repeat_penalty=1.2,
)


from langchain_community.vectorstores import FAISS
# vector_store = FAISS.load_local("faiss_index_document_programmer_guide", local_embeddings, allow_dangerous_deserialization=True)
vector_store = FAISS.load_local("faiss_index_document_GDM", local_embeddings, allow_dangerous_deserialization=True)
# vector_store = InMemoryVectorStore.from_documents(pdf_documents, local_embeddings)

# from langchain.vectorstores import FAISS
# vector_store = FAISS.load_local("faiss_index", local_embeddings, allow_dangerous_deserialization=True)
# vector_store = FAISS.from_documents(documents=pages, embedding=local_embeddings)
# vector_store.save_local("faiss_index")

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

RAG_TEMPLATE = """
##問題(Question):"{question}"

##檢索上下文(Retrieved Context):{context}

##指導方針：
你是一位專精在妊娠期糖尿病的營養師及藥劑師，能在保持技術準確性的同時，以對話回應問題。
你的任務是當醫生向你提問時，根據檢索上下文中的資訊回答醫生的問題。
檢索上下文當中含有範例問題及回答，你必須對範例回答稍加修飾後輸出。

你必須嚴格遵循以下指導方針：
**你只能根據檢索上下文中的資訊回答問題。
**注意回答的篇幅，不需要超過100字。
**針對問題直接提供問題的解答，避免任何不必要的資訊。
**在使用專有名詞時，必須採用和問題中一樣的說法。
**確保回答內容清晰且精確，並保持專業語氣。
**避免重複，並優先確保內容連貫，清晰地解釋觀點。
**當問題中有出現個別病患的狀況時，必須先總結患者的狀況或該狀況所代表的涵義。
**當沒有明確指示時，你可以假設問題中出現的患者指的是妊娠期糖尿病患者。
**你說話的對象是醫生，所以你的回應不能要求要與營養師或醫師討論。
**使用繁體中文回答。

根據以上資訊，請回答：{question}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# retriever = vector_store.as_retriever()
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3"
                                  )
compressor = CrossEncoderReranker(model=reranker, top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

qa_chain = (
    # {"context": retriever | format_docs, "question": RunnablePassthrough()}
    {"context": compression_retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

# related question generate
import re
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

related_question_system_prompt = SystemMessagePromptTemplate.from_template(
"""
You are an assistant specializing in question generation, capable of producing multiple relevant and in-depth follow-up questions based on user input. Your goal is to expand the scope of the user’s query and cover various aspects of the related field. Please adhere to the following guidelines:

Guidelines:
Question Expansion: Each follow-up question should be closely related to the user’s input topic but explore the subject from different angles or levels.
Diversity: Avoid generating overly similar questions. Ensure that the follow-up questions demonstrate diversity in approach and focus.
Specificity: The questions should be specific and actionable, helping the user gain a deeper understanding or exploration of the topic.
Quantity: Generate at least five follow-up questions each time.
Examples:
User Input: "2024 Summer Olympics"
Follow-up Questions:

What new events will be introduced in the 2024 Summer Olympics?
What are the specific dates for the opening and closing ceremonies of the 2024 Olympics?
What changes have been made to the schedule of the 2024 Olympics?
What are the specific events for breakdancing in the 2024 Olympics?
Where will the surfing competitions of the 2024 Olympics take place?
User Input: "How to create charts with Python"
Follow-up Questions:

How can you use Python to create scatter plots?
What other visualization libraries are available in Python?
How do you create line charts in Python?
What are the common issues encountered when creating charts in Python?
How can you add titles and axis labels to charts in Python?
Please follow the above rules to generate relevant follow-up questions based on user input. Do not answer the questions.
Respond in the language specified by the user or the language of the query provided.
"""
)




# Combine system and human prompt templates
human_prompt = HumanMessagePromptTemplate.from_template("{user_input}")
prompt = ChatPromptTemplate.from_messages([related_question_system_prompt, human_prompt])

# Run the LangChain chain
# from langchain.chains import LLMChain
# chain = LLMChain(llm=model, prompt=prompt)
related_question_chain = prompt | model

