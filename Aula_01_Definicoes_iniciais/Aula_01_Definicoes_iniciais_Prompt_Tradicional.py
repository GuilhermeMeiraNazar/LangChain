import os
from pathlib import Path
from dotenv import load_dotenv

# Importações necessárias do LangChain
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate

# 1. Configuração do Caminho e Carregamento do arquivo .env
# Isso garante que o Python ache o .env na pasta raiz do projeto
raiz = Path(__file__).parent.parent
dotenv_path = raiz / '.env'
load_dotenv(dotenv_path=dotenv_path)

# 2. Inicialização do Modelo (LLM)
# Importante: Usamos ChatOpenAI para compatibilidade com o Llama do LM Studio
llm = ChatOpenAI(
    base_url=os.environ.get('LM_STUDIO_URL'),
    api_key=os.environ.get("API_KEY"), 
    model="meta-llama-3.1-8b-instruct",
    temperature=0.7
)

# 3. Definição do Prompt Template
# O {pergunta} é uma variável que será preenchida depois
pergunta = "O que é LangChain e para que serve?"
prompt_tradicional = ChatPromptTemplate.from_template(
    "Responda a seguinte pergunta de forma concisa:\n\n{pergunta}"
)

chain_tradicional = prompt_tradicional | llm

respota_tradicional = chain_tradicional.invoke({"pergunta": pergunta})

print(respota_tradicional.content)