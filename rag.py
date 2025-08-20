from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 1) Charger des pages web (sans chevrons) + User-Agent ---
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) King-RAG/1.0"

loader = WebBaseLoader(web_paths=urls, requests_kwargs={"headers": {"User-Agent": UA}})
docs = loader.load()

# --- 2) Split des documents (sans tiktoken pour éviter une dépendance en plus) ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
doc_splits = splitter.split_documents(docs)

# --- 3) Embeddings 100% locaux via Ollama + Vector store persistant (Chroma) ---
emb = OllamaEmbeddings(model="nomic-embed-text")      # Assure-toi d'avoir: ollama pull nomic-embed-text
vectorstore = Chroma.from_documents(doc_splits, emb, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(k=4)

print(doc_splits)


# --- 4) (Option) Génération avec LLM local via Ollama ---
llm = ChatOllama(model="gemma3:1b", temperature=0)  # Assure-toi d'avoir: ollama pull llama3.1:8b
prompt = ChatPromptTemplate.from_messages([
    ("system", "Réponds STRICTEMENT à partir du contexte. Si l'information manque, dis-le. Termine par une courte liste de sources."),
    ("human", "Contexte:\n{context}\n\nQuestion: {question}"),
])

chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    # Petit test: affichage du nombre de chunks et réponse RAG
    print(f"{len(docs)} pages chargées → {len(doc_splits)} chunks indexés.")
    question = "Donne 3 idées clés sur le prompt engineering."
    print("\nRéponse RAG:\n")
    print(chain.invoke(question))
