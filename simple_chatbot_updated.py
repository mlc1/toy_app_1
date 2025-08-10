import os
from dotenv import load_dotenv, find_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import FileChatMessageHistory

# Cargar variables de entorno desde el archivo .env
# Esta función busca un archivo .env en el directorio actual o en directorios superiores
_ = load_dotenv(find_dotenv())
# Obtenemos la clave de API de OpenAI desde las variables de entorno
# Esta clave es necesaria para autenticarnos con los servicios de OpenAI
#openai_api_key = os.environ["OPENAI_API_KEY"]

# Inicializar el chatbot con el modelo gpt-3.5-turbo de OpenAI
# Esto crea una instancia que nos permite comunicarnos con la API de ChatGPT
chatbot = ChatOllama(model='llama3:latest')

# Definir la plantilla de prompt para nuestras conversaciones
# Esta plantilla determina cómo se estructurarán los mensajes enviados al modelo
prompt = ChatPromptTemplate.from_messages([
    # MessagesPlaceholder reserva un espacio para el historial de conversación anterior
    MessagesPlaceholder(variable_name="history"),
    # Esta tupla define un mensaje del usuario (input) que será reemplazado en cada invocación
    ("human", "{input}")
])

# Crear una cadena ejecutable combinando la plantilla y el chatbot
# El operador '|' conecta estos componentes, similar a una tubería en sistemas Unix
chain = prompt | chatbot

# Función para obtener el historial de chat basado en el ID de sesión
# Esto nos permite tener conversaciones separadas para diferentes usuarios
def get_session_history(session_id: str):
    # Creamos un archivo JSON único para cada usuario basado en su ID de sesión
    # Esto permite mantener conversaciones independientes para cada usuario
    return FileChatMessageHistory(f"messages_{session_id}.json")

# Crear un ejecutable con capacidad de historial de mensajes
# RunnableWithMessageHistory envuelve nuestra cadena y le añade la capacidad de recordar conversaciones
runnable = RunnableWithMessageHistory(
    chain,  # La cadena que procesará los mensajes
    get_session_history,  # Función que proporciona acceso al historial basado en session_id
    input_messages_key="input",  # Nombre de la clave que contiene el mensaje de entrada
    history_messages_key="history"  # Nombre de la clave donde se almacena el historial
)

# Interactuar con el chatbot - Primera interacción
# Le informamos al chatbot sobre nuestro color favorito
response = runnable.invoke(
    {"input": "Mi color favorito es el azul."},  # Mensaje que enviamos al chatbot
    config={"configurable": {"session_id": "usuario1"}}  # Configuración que identifica la sesión
)
# Imprimimos separadores para hacer la salida más legible
print("\n----------\n")
# Mostramos el mensaje que enviamos
print("Mi color favorito es el azul.")
print("\n----------\n")
# Mostramos la respuesta del chatbot
print(response.content)
print("\n----------\n")

# Segunda interacción - Probamos si el chatbot recuerda nuestra preferencia de color
response = runnable.invoke(
    {"input": "¿Cuál es mi color favorito?"},
    config={"configurable": {"session_id": "usuario1"}}
)
print("\n----------\n")
print("¿Cuál es mi color favorito?")
print("\n----------\n")
print(response.content)
print("\n----------\n")

# Continuar la conversación - Tercera interacción con un saludo
response = runnable.invoke(
    {"input": "¡Hola!"},
    config={"configurable": {"session_id": "usuario1"}}
)
print("\n----------\n")
print("¡Hola!")
print("\n----------\n")
print(response.content)
print("\n----------\n")

# Cuarta interacción - Le decimos nuestro nombre al chatbot
response = runnable.invoke(
    {"input": "Mi nombre es Manuel"},
    config={"configurable": {"session_id": "usuario1"}}
)
print("\n----------\n")
print("Mi nombre es Julio")
print("\n----------\n")
print(response.content)
print("\n----------\n")

# Quinta interacción - Comprobamos si el chatbot recuerda nuestro nombre
# Esto demuestra que el sistema de memoria funciona correctamente
response = runnable.invoke(
    {"input": "¿Cómo me llamo?"},
    config={"configurable": {"session_id": "usuario1"}}
)
print("\n----------\n")
print("¿Cómo me llamo?")
print("\n----------\n")
print(response.content)
print("\n----------\n")