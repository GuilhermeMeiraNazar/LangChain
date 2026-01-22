from pathlib import Path # Import movido para o topo
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# 1. Configura os caminhos (Agora o Path já foi importado)
raiz = Path(__file__).parent.parent
dotenv_path = raiz / '.env'
load_dotenv(dotenv_path=dotenv_path)

# 2. Inicializa os Embeddings
embeddings = OpenAIEmbeddings(
    model="nomic-embed-text-v1.5", 
    openai_api_base=os.environ.get('LM_STUDIO_URL'), 
    openai_api_key=os.environ.get("API_KEY"),
    # Adicionamos o parâmetro abaixo para evitar o erro 400 em alguns servidores locais
    check_embedding_ctx_length=False 
)

# 3. Execução
resultado = embeddings.embed_query("LangChain é uma biblioteca para construir aplicações com LLMs")
print(resultado)