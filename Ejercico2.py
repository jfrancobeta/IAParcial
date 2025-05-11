import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Configurar las rutas donde se almacenarán los modelos descargados para optimizar la carga.
os.environ["TRANSFORMERS_CACHE"] = "./models_cache"
os.environ["HF_HOME"] = "./huggingface"

def cargar_modelo(nombre_modelo):
    """
    Carga un modelo pre-entrenado y su tokenizador correspondiente.
    """
    print(f"Cargando modelo: {nombre_modelo}")
    
    tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
    # Añadir token de padding si no existe
    if tokenizador.pad_token is None:
        tokenizador.pad_token = tokenizador.eos_token  # Usar EOS como PAD
    
    modelo = AutoModelForCausalLM.from_pretrained(
        nombre_modelo,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    modelo.eval()
    return modelo, tokenizador

def verificar_dispositivo():
    """
    Verifica si el sistema tiene una GPU disponible o usa la CPU.
    
    Returns:
        torch.device: El dispositivo a utilizar para la inferencia
    """
    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")
        # Mostramos detalles de la GPU si está disponible
        print("GPU disponible. Información:")
        print(f"  Nombre: {torch.cuda.get_device_name(0)}")
        print(f"  Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  Capacidad de cómputo: {torch.cuda.get_device_capability(0)}")
        print(f"  Número de GPUs: {torch.cuda.device_count()}")
    elif torch.backends.mps.is_available():  # Apple Silicon (MPS)
        dispositivo = torch.device("mps")
        print("Utilizando Apple Silicon MPS para la inferencia.")
    else:
        dispositivo = torch.device("cpu")
        print("No se encontró GPU, usando CPU.")
    
    return dispositivo

def generar_texto(modelo, tokenizador, prompt, max_length=100):
    """
    Genera texto a partir de un prompt usando el modelo y tokenizador.
    
    Args:
        modelo: El modelo pre-entrenado cargado
        tokenizador: Tokenizador del modelo
        prompt (str): Texto inicial para la generación
        max_length (int): Longitud máxima del texto generado
    
    Returns:
        str: Texto generado por el modelo
    """
    # Tokenizamos el prompt y lo preparamos para el modelo
    inputs = tokenizador(prompt, return_tensors="pt")
    
    # Movemos los datos de entrada al mismo dispositivo que el modelo
    inputs = {k: v.to(modelo.device) for k, v in inputs.items()}
    
    # Generamos el texto sin necesidad de calcular gradientes (aumentamos la eficiencia)
    with torch.no_grad():
        output = modelo.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,            # Hacemos la generación más variada
            top_p=0.95,                # Controlamos la diversidad con top-p sampling
            temperature=0.7,           # Ajustamos la "creatividad" de la salida
            num_return_sequences=1    # Generamos una única secuencia
        )
    
    # Decodificamos los tokens generados de vuelta a texto
    texto_generado = tokenizador.decode(output[0], skip_special_tokens=True)
    return texto_generado
def preprocesar_entrada(texto, tokenizador, longitud_maxima=512):
    """
    Preprocesa el texto de entrada incluyendo la máscara de atención.
    """
    inputs = tokenizador(
        texto,
        max_length=longitud_maxima,
        truncation=True,
        padding="max_length",  # Añadir padding consistente
        return_tensors="pt"
    )
    return inputs

def generar_respuesta(modelo, entrada_procesada, tokenizador, parametros_generacion=None):
    """
    Genera respuesta incluyendo el manejo correcto de la máscara de atención.
    """
    if parametros_generacion is None:
        parametros_generacion = {
            "max_new_tokens": 100,  # Mejor usar max_new_tokens para controlar respuesta
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 0.7,
            "num_return_sequences": 1,
            "pad_token_id": tokenizador.eos_token_id
        }
    
    # Mover todos los elementos al dispositivo correcto
    inputs = {k: v.to(modelo.device) for k, v in entrada_procesada.items()}
    
    with torch.no_grad():
        salida = modelo.generate(
            **inputs,
            **parametros_generacion
        )
    
    # Decodificar omitiendo los tokens especiales
    respuesta = tokenizador.decode(
        salida[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
    return respuesta

def crear_prompt_sistema(instrucciones):
    """
    Crea un prompt de sistema para dar instrucciones al modelo.
    
    Args:
        instrucciones (str): Instrucciones sobre cómo debe comportarse el chatbot
    
    Returns:
        str: Prompt formateado
    """
    # Formato tipo chatbot con instrucciones de comportamiento
    prompt = (
        f"Eres un asistente IA con las siguientes instrucciones: {instrucciones}\n"
        "### Historial de conversación:\n"
        "[Usuario]: "
    )
    return prompt

def interaccion_simple():
    # Cargamos modelo y tokenizador
    modelo, tokenizador = cargar_modelo("gpt2")
    
    # Creamos prompt de sistema con personalidad específica
    instrucciones = (
        "responde de manera educada y profesional. "
        "Proporciona explicaciones detalladas pero concisas."
    )
    prompt_sistema = crear_prompt_sistema(instrucciones)
    
    # Simulamos una entrada de usuario
    entrada_usuario = "¿Cómo puedo mejorar mis habilidades en programación?"
    
    # Combinamos el prompt del sistema con la entrada del usuario
    prompt_completo = prompt_sistema + entrada_usuario
    
    # Preprocesamos la entrada
    entrada_procesada = preprocesar_entrada(prompt_completo, tokenizador)
    
    # Generamos la respuesta
    respuesta = generar_respuesta(modelo, entrada_procesada, tokenizador)
    
    # Mostramos resultados
    print("\n--- Interacción completa ---")
    print(f"Prompt del sistema:\n{prompt_sistema}")
    print(f"Entrada del usuario:\n{entrada_usuario}")
    print(f"\nRespuesta generada:\n{respuesta}")

if __name__ == "__main__":
    interaccion_simple()