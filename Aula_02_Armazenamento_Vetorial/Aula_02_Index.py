from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# --- CORREÇÃO 1: Importar o Document oficial do LangChain ---
from langchain_core.documents import Document 
import faiss

# 1. Configura os caminhos
raiz = Path(__file__).parent.parent
dotenv_path = raiz / '.env'
load_dotenv(dotenv_path=dotenv_path)

# --- CORREÇÃO 2: Remova a "class Document" manual que estava aqui ---

# 2. Criando a lista de documentos da empresa
# O uso permanece igual, pois o Document do LangChain aceita page_content e metadata
documentos_empresa = [
    Document(
        page_content="Política de Férias: O colaborador deve solicitar suas férias com 30 dias de antecedência. O período mínimo de gozo é de 10 dias corridos.",
        metadata={"tipo": "RH", "departamento": "Recursos Humanos", "ano": 2024, "id_doc": 101}
    ),
    Document(
        page_content="Processo de Reembolso: Para despesas de viagem, anexe todos os comprovantes no sistema interno em até 5 dias úteis após o retorno.",
        metadata={"tipo": "Financeiro", "departamento": "Financeiro", "ano": 2024, "id_doc": 102}
    ),
    Document(
        page_content="Guia de TI - VPN: Para acessar a rede interna, utilize o cliente AnyConnect com suas credenciais de rede e autenticação de dois fatores (MFA).",
        metadata={"tipo": "TI", "departamento": "Tecnologia", "ano": 2023, "id_doc": 103}
    ),
    Document(
        page_content="Código de Ética: É dever de todo funcionário agir com transparência, respeito e integridade, evitando conflitos de interesse.",
        metadata={"tipo": "Compliance", "departamento": "Jurídico", "ano": 2024, "id_doc": 104}
    )
]

# 3. Inicializa os Embeddings
embeddings = OpenAIEmbeddings(
    model="nomic-embed-text-v1.5", 
    openai_api_base=os.environ.get('LM_STUDIO_URL'), 
    openai_api_key=os.environ.get("API_KEY"),
    check_embedding_ctx_length=False 
)

# 4. Criando o banco vetorial
# Nota: O LangChain gerencia o index do FAISS internamente quando usamos .from_documents
faiss_db = FAISS.from_documents(documentos_empresa, embeddings) # FAISS.from_documents: Transforma os textos em vetores numéricos usando o modelo de embedding e organiza esses vetores em um índice (banco de dados) para buscas rápidas.

pergunta = "Minha vpn não conecta, o que fazer?"

resultado = faiss_db.similarity_search(pergunta, k=1) #similarity_search: Converte a pergunta em vetor e calcula a distância matemática no espaço vetorial. O k=1 ordena os resultados e retorna apenas o documento mais próximo (mais similar).

print(f"Pergunta: {pergunta}") 
for doc in resultado:
    print(f"Resultado: {doc.page_content} (Metadados: {doc.metadata})")