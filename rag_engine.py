import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
TMP_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="PO Assistant - Crie histórias de usuário", page_icon="🕵🏻")

st.title("PO Assistant - Crie histórias de usuário!")
st.markdown("A ferramenta utiliza RAG (Retrieval-Augmented Generation) para transformar sua base de conhecimento em um assistente de Product Owner especializado. Faça upload de documentações, requisitos ou contextos do projeto para receber sugestões contextualizadas para criar user stories!")
st.markdown("Criado por Victor Augusto Souza Resende")

def load_documents():
    """
    Carrega todos os PDFs do diretório TMP_DIR usando PyPDFLoader.

    Retorna
    -------
    List[Document]
        Lista de objetos Document resultantes da leitura e divisão (chunks) de cada PDF.
    """
    all_docs = []
    for pdf_file in TMP_DIR.glob("*.pdf"):
        loader = PyPDFLoader(pdf_file.as_posix())
        pdf_docs = loader.load_and_split()
        all_docs.extend(pdf_docs)
    return all_docs

def build_vectorstore(documents):
    """
    Cria um índice vetorial FAISS em memória a partir dos documentos e retorna um retriever.

    Parâmetros
    ----------
    documents : List[Document]
        Lista de documentos que serão convertidos em embeddings e indexados.

    Retorna
    -------
    FAISS
        Objeto retriever do FAISS configurado para buscar os documentos mais relevantes (k=7).
    """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(documents, embeddings)
    return vectordb.as_retriever(search_kwargs={'k': 7})

def format_chat_history(messages):
    """
    Converte o histórico de mensagens (lista de tuplas) em uma string para injetar no prompt.

    Parâmetros
    ----------
    messages : List[Tuple[str, str]]
        Histórico de mensagens, onde cada tupla contém (mensagem_do_usuário, mensagem_do_assistente).

    Retorna
    -------
    str
        Texto formatado contendo todo o histórico de conversas.
    """
    history_str = ""
    for i, (user_msg, ai_msg) in enumerate(messages):
        history_str += f"Usuário: {user_msg}\n"
        history_str += f"Assistente: {ai_msg}\n"
    return history_str.strip()

def build_rices_prompt(context, chat_history, user_query):
    """
    Monta o prompt no estilo RICES para cada mensagem.

    Parâmetros
    ----------
    context : str
        Contexto adicional e trechos relevantes dos documentos.
    chat_history : str
        Histórico de conversas formatado.
    user_query : str
        Pergunta realizada pelo usuário.

    Retorna
    -------
    str
        Texto final do prompt pronto para ser utilizado no modelo.
    """
    prompt = f"""
    **Role**
    You are an experienced Assistant Product Owner, specialized in transforming business requirements into well-structured user stories. Your role is to analyze the context provided through relevant documents and information, and create user stories that follow agile best practices.
    
    **Instruction**
    Answer clearly, precisely and professionally, adapting your tone to the context of the question. Always write the user story with the IT technical needs, clearly explained for developers, data scientists or something related with the needs.

    **Context**
    To help with the creation of the user story, use the context of the following project documentation:
    {context}

    **Examples**
    Q: Customers need an easy way to track their orders and receive real-time updates on delivery status.
    A: I as an e-commerce customer, want to receive real-time updates on the status of my order so that I can schedule it and be more confident about the delivery. 

    **Attention**
    - Always follow the template of: Eu como usuário (author of the action - persona) quero (desired functionality) para (added value to the desired functionality).
    - Considers stakeholder needs, aligns stories with product strategy. 
    - Every user story should contain an explicit requirement for at least 3 items for technical metrics, acceptance criteria (or requirements) and risks.
    - Always write the user story with the IT technical needs, clearly explained for dev, data science or something related with the needs.
    - Always answer in Portuguese.

    Chat history: 
    {chat_history}

    Question: 
    {user_query}
    """

    return prompt.strip()

def run_rices_query(retriever, user_query, context):
    """
    Executa a query usando o estilo de prompt RICES e retorna a resposta.

    Esta função:
    1) Formata o histórico de chat
    2) Recupera documentos relevantes (FAISS)
    3) Concatena o 'context' adicional do usuário com os trechos relevantes
    4) Constrói o prompt RICES
    5) Chama ChatOpenAI e retorna a resposta

    Parâmetros
    ----------
    retriever : FAISS
        Objeto retriever para buscar documentos relevantes no índice FAISS.
    user_query : str
        Pergunta do usuário.
    context : str
        Contexto adicional fornecido pelo usuário.
    
    Retorna
    -------
    str
        Resposta gerada pelo modelo ChatOpenAI.
    """
    chat_history_str = format_chat_history(st.session_state.messages)
    docs = retriever.get_relevant_documents(user_query)
    docs_text = "\n".join([doc.page_content for doc in docs])
    context_final = f"{context}[Documentos Relevantes]\n{docs_text}"
    final_prompt = build_rices_prompt(
        context=context_final,
        chat_history=chat_history_str,
        user_query=user_query
    )
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",
        temperature=0.2
    )
    response = llm.predict(final_prompt)
    st.session_state.messages.append((user_query, response))
    return response

def process_documents():
    """
    Processa os PDFs enviados, cria o índice FAISS e armazena o retriever em session_state.

    Esta função:
    1) Salva PDFs enviados no diretório temporário TMP_DIR
    2) Cria o índice FAISS e o armazena em st.session_state.retriever
    3) Limpa o TMP_DIR após o processo

    Retorna
    -------
    None
    """
    if not st.session_state.source_docs:
        st.warning("Por favor, envie ao menos um arquivo PDF.")
        return

    for uploaded_file in st.session_state.source_docs:
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=TMP_DIR.as_posix(),
            suffix=".pdf"
        ) as tmp_file:
            tmp_file.write(uploaded_file.read())

    try:
        documents = load_documents()
        for f in TMP_DIR.iterdir():
            f.unlink()
        st.session_state.retriever = build_vectorstore(documents)
        st.success("Índice FAISS criado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao processar documentos: {e} ")

def boot():
    """
    Função principal da aplicação Streamlit.

    Inicializa o estado do chat, define a barra lateral para upload e processamento
    de documentos, exibe o histórico de mensagens e gerencia a interação
    com o usuário via campo de input.

    Retorna
    -------
    None
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.title("Documentações e contextos")
        st.session_state.source_docs = st.file_uploader(
            "Envie PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )
        st.button("Processar Documentos ✨", on_click=process_documents)

    for user_msg, ai_msg in st.session_state.messages:
        st.chat_message("human").write(user_msg)
        st.chat_message("ai").write(ai_msg)

    if st.session_state.get("retriever"):
        with st.sidebar:
            user_context = st.text_area(
                "Contexto adicional (opcional)",
                placeholder="Se houver algum detalhe, cole aqui. Ex: data de referência, setor específico etc."
            )
        if user_query := st.chat_input("Digite sua pergunta aqui..."):
            st.chat_message("human").write(user_query)
            response = run_rices_query(
                retriever=st.session_state.retriever,
                user_query=user_query,
                context=user_context
            )
            st.chat_message("ai").write(response)
    else:
        st.info("Faça upload dos PDFs no lado esquerdo e clique em 'Processar Documentos' para começar.")

if __name__ == "__main__":
    boot()
