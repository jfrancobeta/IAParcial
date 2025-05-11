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


class GestorContexto:
    """
    Clase para gestionar el contexto de una conversación con el chatbot.
    """
    
    def __init__(self, longitud_maxima=1024, formato_mensaje=None):
        self.historial = []
        self.longitud_maxima = longitud_maxima
        self.formato_mensaje = formato_mensaje or self._formato_predeterminado
        self.mensaje_sistema = None

    def _formato_predeterminado(self, rol, contenido):
        """
        Formato predeterminado para mensajes usando marcas de conversación.
        """
        if rol == 'sistema':
            return f"### Instrucciones del Sistema:\n{contenido}"
        return f"[{rol.capitalize()}]: {contenido}"

    def agregar_mensaje(self, rol, contenido):
        """
        Agrega un mensaje al historial manteniendo estructura de conversación.
        """
        if rol == 'sistema':
            self.mensaje_sistema = contenido
            return
            
        mensaje_formateado = self.formato_mensaje(rol, contenido)
        self.historial.append((rol, mensaje_formateado))

    def construir_prompt_completo(self):
        """
        Construye el prompt completo con contexto histórico.
        """
        componentes = []
        if self.mensaje_sistema:
            componentes.append(self.formato_mensaje('sistema', self.mensaje_sistema))
        
        for rol, mensaje in self.historial:
            componentes.append(mensaje)
        
        return "\n".join(componentes)

    def truncar_historial(self, tokenizador):
        """
        Trunca el historial manteniendo el contexto más relevante.
        """
        total_tokens = 0
        nuevo_historial = []
        
        # Primero añadimos el mensaje del sistema si existe
        if self.mensaje_sistema:
            sistema_tokens = tokenizador.tokenize(self.formato_mensaje('sistema', self.mensaje_sistema))
            total_tokens += len(sistema_tokens)
        
        # Añadimos mensajes desde el más reciente al más antiguo
        for rol, mensaje in reversed(self.historial):
            mensaje_tokens = tokenizador.tokenize(mensaje)
            if total_tokens + len(mensaje_tokens) > self.longitud_maxima:
                break
            total_tokens += len(mensaje_tokens)
            nuevo_historial.insert(0, (rol, mensaje))  # Mantener orden original
        
        self.historial = nuevo_historial

class Chatbot:
    """
    Implementación de chatbot con manejo avanzado de contexto.
    """
    
    def __init__(self, modelo_id, instrucciones_sistema=None):
        self.modelo, self.tokenizador = cargar_modelo(modelo_id)
        self.dispositivo = verificar_dispositivo()
        self.gestor_contexto = GestorContexto()
        
        if instrucciones_sistema:
            self.gestor_contexto.agregar_mensaje('sistema', instrucciones_sistema)

    def responder(self, mensaje_usuario, parametros_generacion=None):
        # Agregar mensaje del usuario al contexto
        self.gestor_contexto.agregar_mensaje('usuario', mensaje_usuario)
        
        # Construir y truncar el prompt
        self.gestor_contexto.truncar_historial(self.tokenizador)
        prompt = self.gestor_contexto.construir_prompt_completo()
        
        # Preprocesar y generar respuesta
        entrada_procesada = preprocesar_entrada(prompt, self.tokenizador)
        respuesta = generar_respuesta(self.modelo, entrada_procesada, self.tokenizador, parametros_generacion)
        
        # Agregar respuesta al contexto y devolver
        self.gestor_contexto.agregar_mensaje('asistente', respuesta)
        return respuesta
    
def prueba_conversacion():
    # Configuración del chatbot
    instrucciones = (
        "Eres un asistente útil especializado en tecnología. "
        "Proporciona respuestas claras y técnicas. "
        "Mantén las explicaciones en menos de 3 párrafos."
    )
    
    chatbot = Chatbot("gpt2", instrucciones_sistema=instrucciones)
    
    # Simulación de conversación
    turnos = [
        "¿Qué es un Transformer en machine learning?",
        "¿Y cómo funciona el mecanismo de atención?",
        "¿Qué ventajas tiene sobre los modelos RNN tradicionales?",
        "Dame un ejemplo de implementación práctica"
    ]
    
    for turno in turnos:
        print(f"\n[Usuario]: {turno}")
        respuesta = chatbot.responder(turno)
        print(f"\n[Asistente]: {respuesta}\n")
        
    # Mostrar contexto final
    print("\nContexto completo de la conversación:")
    print(chatbot.gestor_contexto.construir_prompt_completo())

if __name__ == "__main__":
    prueba_conversacion()