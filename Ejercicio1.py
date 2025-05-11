import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Configurar las rutas donde se almacenarán los modelos descargados para optimizar la carga.
os.environ["TRANSFORMERS_CACHE"] = "./models_cache"
os.environ["HF_HOME"] = "./huggingface"

def cargar_modelo(nombre_modelo):
    """
    Carga un modelo pre-entrenado y su tokenizador correspondiente.
    
    Args:
        nombre_modelo (str): Identificador del modelo en Hugging Face Hub
    
    Returns:
        tuple: (modelo, tokenizador)
    """
    print(f"Cargando modelo: {nombre_modelo}")
    
    # Cargamos el tokenizador desde Hugging Face para el modelo especificado
    tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
    
    # Configuramos el modelo para que utilice menos memoria y sea más rápido
    modelo = AutoModelForCausalLM.from_pretrained(
        nombre_modelo,
        torch_dtype=torch.float16,  # Usamos half-precision para ahorrar memoria
        low_cpu_mem_usage=True,     # Reducción de uso de memoria en la CPU
        device_map="auto"           # Permitir que el modelo se distribuya automáticamente entre dispositivos
    )
    
    # Ponemos el modelo en modo evaluación (desactiva la actualización de pesos)
    modelo.eval()  # Modo inferencia
    
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

# Función principal para probar el modelo
def main():
    # Verificamos el dispositivo para determinar si usamos GPU o CPU
    dispositivo = verificar_dispositivo()
    print(f"Dispositivo en uso: {dispositivo}")
    
    # Elegimos el modelo pre-entrenado GPT-2 para probar la generación de texto
    nombre_modelo = "gpt2"  # También se pueden probar otros modelos como distilgpt2
    modelo, tokenizador = cargar_modelo(nombre_modelo)
    
    # Definimos el prompt que se usará para la generación
    prompt = "En un mundo dominado por la inteligencia artificial,"
    print("\nPrompt:", prompt)
    
    # Generamos el texto utilizando el modelo y mostramos la salida
    texto_generado = generar_texto(modelo, tokenizador, prompt, max_length=150)
    print("\nTexto generado:")
    print(texto_generado)
    
    # Mostramos información adicional sobre el modelo
    parametros_totales = sum(p.numel() for p in modelo.parameters())
    print(f"\nInformación del modelo:")
    print(f"  Nombre del modelo: {nombre_modelo}")
    print(f"  Número de parámetros: {parametros_totales/1e6:.2f} millones")

if __name__ == "__main__":
    main()
