import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Diretório temporário para processar arquivos
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
TMP_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="PO Assistant - Gere artefatos!", page_icon="🕵🏻")

# st.markdown(
#     """
#     <style>
#     .e1dbuyne8, .ekr3hml1 {
#         background: url(https://www.aprovaconcursos.com.br/noticias/wp-content/uploads/2021/09/Caixa-economica.jpg)
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
#     [theme]
# primaryColor="#f1f1f1"
# backgroundColor="#4775a2"
# secondaryBackgroundColor="#1e4881"
# textColor="#f3eded"
# )

st.title("PO Assistant - Gere artefatos!")
with st.expander("Entenda mais!"):
    st.markdown("""
    A ferramenta utiliza RAG (Retrieval-Augmented Generation) para 
    transformar sua base de conhecimento em um assistente de Product Owner especializado. 

    Faça upload de documentações, requisitos ou contextos do projeto para receber 
    sugestões contextualizadas para criar user stories!

    Quer dar sugestões de melhoria? Envie um email para: victor.resende@caixa.gov.br
    """)
st.write("Criado por Victor Augusto Souza Resende")


def chunk_pdf(file_path):
    """
    Carrega um arquivo PDF e realiza chunking usando RecursiveCharacterTextSplitter.

    Esta função lê o PDF indicado por `file_path`, carrega seu conteúdo usando
    `PyPDFLoader` e, em seguida, divide o texto em blocos (chunks) menores
    por meio do `RecursiveCharacterTextSplitter`.

    Parameters
    ----------
    file_path : str
        Caminho completo para o arquivo PDF a ser processado.

    Returns
    -------
    list of langchain.docstore.document.Document
        Uma lista de objetos Document, cada um contendo parte do texto do PDF.
    """
    loader = PyPDFLoader(file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    pdf_docs = loader.load_and_split(text_splitter=text_splitter)
    return pdf_docs


def chunk_text(file_path):
    """
    Carrega um arquivo de texto (TXT ou VTT) e realiza chunking com RecursiveCharacterTextSplitter.

    Esta função lê o arquivo indicado por `file_path`, carrega seu conteúdo usando
    `TextLoader` e, em seguida, divide o texto em blocos (chunks) menores
    por meio do `RecursiveCharacterTextSplitter`.

    Parameters
    ----------
    file_path : str
        Caminho completo para o arquivo de texto a ser processado.

    Returns
    -------
    list of langchain.docstore.document.Document
        Lista de objetos Document, cada um com parte do texto carregado.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)


def load_documents():
    """
    Carrega todos os arquivos armazenados em TMP_DIR e realiza chunking de cada um.

    Percorre o diretório temporário `TMP_DIR`, identifica a extensão do arquivo,
    usa o loader apropriado (PDF ou texto) e, em seguida, chama as funções de chunking.
    Documentos de extensões desconhecidas são ignorados.

    Returns
    -------
    list of langchain.docstore.document.Document
        Lista de objetos Document com o conteúdo já dividido em chunks.
    """
    all_docs = []

    for file_path in TMP_DIR.glob("*"):
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            pdf_docs = chunk_pdf(file_path.as_posix())
            all_docs.extend(pdf_docs)
        elif suffix in [".txt", ".vtt"]:
            txt_docs = chunk_text(file_path.as_posix())
            all_docs.extend(txt_docs)
        else:
            pass

    return all_docs


def build_vectorstore(documents):
    """
    Cria um índice vetorial FAISS a partir de uma lista de documentos.

    Converte cada documento em embeddings usando `OpenAIEmbeddings` e,
    em seguida, cria um índice FAISS. Retorna um objeto retriever para facilitar
    as buscas de documentos relevantes.

    Parameters
    ----------
    documents : list of langchain.docstore.document.Document
        Lista de documentos (com conteúdo e metadados) para indexar.

    Returns
    -------
    langchain.vectorstores.faiss.FAISS
        Objeto FAISS configurado como retriever, permitindo consultas
        por similaridade.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(documents, embeddings)
    return vectordb.as_retriever(search_kwargs={'k': 3})


def format_chat_history(messages):
    """
    Converte o histórico de mensagens em uma string formatada.

    Cada item da lista de mensagens é uma tupla (mensagem_do_usuário, mensagem_do_assistente).
    O retorno é uma string contendo essas mensagens de modo linear.

    Parameters
    ----------
    messages : list of tuple
        Lista de tuplas (user_message, ai_message) representando o histórico do chat.

    Returns
    -------
    str
        Texto formatado contendo todo o histórico de conversas.
    """
    history_str = ""
    for i, (user_msg, ai_msg) in enumerate(messages):
        history_str += f"Usuário: {user_msg}\n"
        history_str += f"Assistente: {ai_msg}\n"
    return history_str.strip()


def build_rices_prompt(context, chat_history, user_query, artifact_format):
    """
    Monta o prompt de consulta no estilo RICES, adaptando o conteúdo ao formato desejado (HU, BDD ou TDD).

    Dependendo do tipo selecionado (`artifact_format`), adiciona instruções específicas
    (por exemplo, formato de história de usuário, Gherkin para BDD ou foco em testes para TDD).

    Parameters
    ----------
    context : str
        Texto que agrupa o contexto do usuário e/ou trechos relevantes dos documentos.
    chat_history : str
        Histórico de conversa formatado.
    user_query : str
        Consulta ou pergunta feita pelo usuário.
    artifact_format : str
        Formato desejado, podendo ser "HU", "BDD" ou "TDD".

    Returns
    -------
    str
        O texto completo do prompt que será enviado ao modelo de linguagem.
    """
    extra_instructions = ""
    if artifact_format == "HU":
        extra_instructions = """
        - Use the format: "Eu como [persona] quero [funcionalidade] para [valor gerado]."
        - **ALWAYS** Include at least 3 technical requirements, 3 acceptance criteria and 3 risks.
        - **Answer Example**
            I, as a salaried worker, want to consult my FGTS balance via the CAIXA Tem app so that I can keep track of my deposits and plan my future use.
            Acceptance criteria
                Scenario: Successful consultation. Given that I am a registered CAIXA Tem user and I have active FGTS when I access the “Consult FGTS” option then I should see the total available balance
        """
    elif artifact_format == "BDD":
        extra_instructions = """
        - Use the Gherkin format: Feature, Scenario, Given/When/Then.
        - **ALWAYS** Consider BDD-style acceptance criteria.
        - **Answer Example**
            Feature: Consultation of FGTS Balance at CAIXA Tem

            Background:
                Given that the user is logged in to the CAIXA Tem application
                And has undergone two-factor authentication

            Scenario: Successful FGTS balance inquiry
                Given that the user has an active FGTS account
                When you select the “Consult FGTS” option
                Then the system displays the total available balance
                And displays the statement for the last 3 months
                And displays information on the last deposit
                And displays available withdrawal options
        """
    elif artifact_format == "TDD":
        extra_instructions = """
        - Focus on test cases and minimum implementation requirements.
        - Explain how each test would be validated.
        """

    prompt = f"""
    **Role**
    You are an experienced Assistant Product Owner, specialized in transforming business requirements into well-structured {artifact_format}. Your role is to analyze the context provided through relevant documents and information, and create {artifact_format} that follow agile best practices.
    
    **Instruction**
    Answer clearly, precisely and professionally, adapting your tone to the context of the question.
    Always tailor the answer in Portuguese, considering the chosen format: {artifact_format}.
    {extra_instructions}

    **Context**
    To help with the creation of the {artifact_format}, use the context of the following project documentation:
    {context}

    **Attention**
    - Considers stakeholder needs, aligns stories with product strategy. 
    - Always write the {artifact_format} with the IT technical needs, clearly explained for dev, data science or something related with the needs.
    - Always answer in Portuguese.

    Chat history: 
    {chat_history}

    Question: 
    {user_query}
    """
    print(prompt.strip())
    return prompt.strip()


def run_rices_query(retriever, user_query, context, artifact_format):
    """
    Executa uma query usando o prompt RICES e retorna a resposta gerada pelo modelo.

    Esta função:
    1. Formata o histórico de conversas.
    2. Recupera documentos relevantes do `retriever`.
    3. Concatena o contexto adicional do usuário com os trechos relevantes.
    4. Constrói o prompt usando `build_rices_prompt`.
    5. Chama o modelo ChatOpenAI para obter a resposta.
    6. Armazena a conversa em `st.session_state.messages`.

    Parameters
    ----------
    retriever : langchain.vectorstores.faiss.FAISS
        Um objeto retriever que permite buscar documentos relevantes no índice FAISS.
    user_query : str
        Pergunta ou comando que o usuário forneceu.
    context : str
        Texto adicional fornecido pelo usuário para complementar a consulta.
    artifact_format : str
        Formato desejado para a resposta ("HU", "BDD" ou "TDD").

    Returns
    -------
    str
        Resposta gerada pelo modelo de linguagem, armazenada no histórico do chat.
    """
    chat_history_str = format_chat_history(st.session_state.messages)
    docs = retriever.get_relevant_documents(user_query)
    docs_text = "\n".join([doc.page_content for doc in docs])
    context_final = f"{context}\n[Documentos Relevantes]\n{docs_text}"

    final_prompt = build_rices_prompt(
        context=context_final,
        chat_history=chat_history_str,
        user_query=user_query,
        artifact_format=artifact_format
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
    Processa arquivos (PDF, TXT, VTT) e/ou texto fornecido pelo usuário, criando um índice FAISS.

    1. Salva os arquivos recebidos via `st.file_uploader` em um diretório temporário.
    2. Carrega e faz chunking dos documentos usando funções auxiliares.
    3. Cria o índice vetorial (FAISS) usando `build_vectorstore`.
    4. Armazena o objeto retriever em `st.session_state.retriever`.

    A função exibe mensagens de status ou erro ao usuário via Streamlit.

    Returns
    -------
    None
        Esta função não retorna valor, mas atualiza `st.session_state.retriever` e exibe status no app.
    """
    all_documents = []

    # Salvando arquivos no TMP_DIR
    if st.session_state.uploaded_files:
        for uploaded_file in st.session_state.uploaded_files:
            suffix = Path(uploaded_file.name).suffix.lower()
            if suffix in [".pdf", ".txt", ".vtt"]:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    dir=TMP_DIR.as_posix(),
                    suffix=suffix
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())

        # Carrega documentos de todos os tipos
        try:
            docs = load_documents()
            all_documents.extend(docs)
            # Limpa o TMP_DIR
            for f in TMP_DIR.iterdir():
                f.unlink()
        except Exception as e:
            st.error(f"Erro ao processar documentos: {e}")

    # Se houver texto manual
    if st.session_state.manual_text:
        text_doc = Document(
            page_content=st.session_state.manual_text,
            metadata={"source": "texto_usuario"}
        )
        all_documents.append(text_doc)

    # Se não houver nenhum documento ou texto, avisa
    if not all_documents:
        st.warning("Não há arquivos ou texto para processar. Insira ao menos um deles.")
        return

    # Constrói o índice (vectorstore)
    try:
        st.session_state.retriever = build_vectorstore(all_documents)
        st.success("Índice FAISS criado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao criar índice FAISS: {e}")


def boot():
    """
    Função principal que inicializa e gerencia a interface Streamlit.

    1. Define variáveis de estado para upload de arquivos, texto, histórico de mensagens, etc.
    2. Renderiza a barra lateral para upload de documentos ou texto.
    3. Cria/atualiza a base de conhecimento (FAISS) quando solicitado.
    4. Mostra o histórico de mensagens no corpo principal.
    5. Exibe um campo de input para o usuário fazer perguntas e recebe as respostas do modelo.

    Returns
    -------
    None
        Esta função não retorna valor; ela desenha a interface via Streamlit e atualiza o estado global.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Variáveis de estado para upload e texto
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = None
    if "manual_text" not in st.session_state:
        st.session_state.manual_text = ""

    # Formato escolhido (HU, BDD, TDD)
    if "artifact_format" not in st.session_state:
        st.session_state.artifact_format = "HU"

    # Define um mini-prompt inicial vazio
    if "mini_prompt" not in st.session_state:
        st.session_state.mini_prompt = ""

    with st.sidebar:
        st.title("Base de conhecimento")

        opcao_envio = st.selectbox(
            "Como deseja fornecer os dados?",
            ("Documentações (PDF, TXT ou VTT)", "Texto aberto", "Ambos")
        )

        # Se o usuário escolher Documentações ou Ambos,
        # exibimos o file_uploader
        if opcao_envio in ("Documentações (PDF, TXT ou VTT)", "Ambos"):
            st.session_state.uploaded_files = st.file_uploader(
                "Envie arquivos (opcional)",
                type=["pdf", "txt", "vtt"],
                accept_multiple_files=True
            )
        else:
            st.session_state.uploaded_files = None

        # Se o usuário escolher Texto aberto ou Ambos,
        # exibimos o text_area
        if opcao_envio in ("Texto aberto", "Ambos"):
            st.session_state.manual_text = st.text_area(
                "E/ou insira um texto (opcional)",
                placeholder="Cole aqui qualquer texto que gostaria de incluir no índice.",
                max_chars=2000
            )
        else:
            st.session_state.manual_text = ""

        if st.button("Criar base de conhecimento ✨"):
            process_documents()

    # Exibe histórico de mensagens
    for user_msg, ai_msg in st.session_state.messages:
        st.chat_message("human").write(user_msg)
        st.chat_message("ai").write(ai_msg)

    if st.session_state.get("retriever"):
        with st.sidebar:
            user_context = st.text_area(
                "Contexto adicional (opcional)",
                placeholder="Algum detalhe extra para essa pergunta específica?",
                max_chars=500
            )

        
        col_artifact, col_action = st.columns(2)

        with col_artifact:
            # Seleciona o formato do artefato (HU, BDD, TDD)
            st.session_state.artifact_format = st.selectbox(
                "Escolha o tipo de artefato que deseja gerar",
                ("HU", "BDD", "TDD")
            )

        with col_action:
            # Novo selectbox: Ação que o usuário deseja realizar
            action_option = st.selectbox(
                "Selecione a ação",
                ["Escrever", "Reescrever", "Pontuar novamente"]
            )

        # Define o mini prompt baseado na ação escolhida
        if action_option == "Escrever":
            st.session_state.mini_prompt = ""
        elif action_option == "Reescrever":
            # Orienta para reescrever
            st.session_state.mini_prompt = (
                f"Think and **rewrite** or **refactor** the {st.session_state.artifact_format} with different ideas: "
            )
        else:
            # Orienta para pontuar novamente
            st.session_state.mini_prompt = (
                f"Following the Fibonacci scale, re-score the {st.session_state.artifact_format}: "
            )

        # Campo de input do usuário. O placeholder recebe o mini_prompt atual.
        user_query = st.chat_input(placeholder=st.session_state.mini_prompt)

        if user_query:
            # Se a ação não for "Escrever", então queremos anexar o user_query ao mini_prompt
            # para que o prompt final inclua ambas as partes.
            if action_option != "Escrever":
                final_query = st.session_state.mini_prompt + user_query
            else:
                final_query = user_query

            # Exibe no chat o que será realmente enviado ao modelo
            st.chat_message("human").write(final_query)

            response = run_rices_query(
                retriever=st.session_state.retriever,
                user_query=final_query,
                context=user_context,
                artifact_format=st.session_state.artifact_format
            )
            st.chat_message("ai").write(response)
    else:
        st.info("Faça upload de arquivos ou insira texto e clique em 'Criar base de conhecimento' para começar.")


if __name__ == "__main__":
    boot()
