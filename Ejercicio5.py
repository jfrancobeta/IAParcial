import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import gradio as gr
from peft import LoraConfig, get_peft_model, TaskType

import platform
from transformers import BitsAndBytesConfig
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

def configurar_cuantizacion(bits=4):
    """
    Configura la cuantización del modelo para diferentes precisiones.
    """
    if platform.system() == "Windows":
        print("Advertencia: Cuantización deshabilitada porque bitsandbytes no es compatible con Windows.")
        return None

    from transformers import BitsAndBytesConfig
    compute_dtype = torch.float16

    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )
    elif bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["lm_head"],
            bnb_8bit_compute_dtype=compute_dtype
        )
    else:
        raise ValueError("Bits deben ser 4 u 8")

def cargar_modelo_optimizado(nombre_modelo, optimizaciones=None):
    """
    Carga el modelo con múltiples optimizaciones aplicadas.
    """
    default_optim = {
        "cuantizacion": True,
        "bits": 4,
        "offload_cpu": False,
        "flash_attention": True,
        "sliding_window": 1024
    }
    optim = default_optim | (optimizaciones or {})
    
    quantization_config = None
    if optim["cuantizacion"]:
        quantization_config = configurar_cuantizacion(optim["bits"])
    
    device_map = "auto" if not optim["offload_cpu"] else {"": "cpu"}
    
    model = AutoModelForCausalLM.from_pretrained(
        nombre_modelo,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.float16,
        use_flash_attention_2=optim["flash_attention"],
        attn_implementation="flash_attention_2" if optim["flash_attention"] else None
    )
    
    if optim["sliding_window"]:
        aplicar_sliding_window(model, optim["sliding_window"])
    
    tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
    return model, tokenizer

def aplicar_sliding_window(modelo, window_size=1024):
    """
    Configura la atención de ventana deslizante en todas las capas.
    """
    for layer in modelo.model.layers:
        if hasattr(layer.self_attn, "sliding_window"):
            layer.self_attn.sliding_window = window_size
        elif hasattr(layer.self_attn, "window_size"):
            layer.self_attn.window_size = window_size

def evaluar_rendimiento(modelo, tokenizador, texto_prueba, dispositivo):
    """
    Evalúa el rendimiento del modelo con métricas clave.
    """
    inputs = tokenizador(texto_prueba, return_tensors="pt").to(dispositivo)
    gen_kwargs = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7
    }
    
    # Calentamiento
    _ = modelo.generate(**inputs, **gen_kwargs)
    
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    outputs = modelo.generate(**inputs, **gen_kwargs)
    
    elapsed = time.time() - start_time
    mem_usage = torch.cuda.max_memory_allocated() / 1e9
    tokens_gen = outputs.shape[1] - inputs.input_ids.shape[1]
    
    return {
        "tiempo": f"{elapsed:.2f}s",
        "memoria": f"{mem_usage:.2f}GB",
        "tokens_por_segundo": f"{tokens_gen / elapsed:.2f} t/s"
    }

def configurar_peft(modelo, r=8, lora_alpha=32):
    """
    Configura fine-tuning adaptativo con LoRA.
    """
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # Módulos comunes en modelos transformer
    )
    return get_peft_model(modelo, config)

def guardar_modelo(modelo, tokenizador, ruta):
    """
    Guarda modelo y tokenizador optimizado para producción.
    """
    # Crear directorio si no existe
    os.makedirs(ruta, exist_ok=True)
    
    # Guardar con nombres de archivo explícitos
    modelo.save_pretrained(ruta, safe_serialization=True)
    tokenizador.save_pretrained(ruta)
    print(f"Modelo y tokenizador guardados en: {ruta}")

def cargar_modelo_personalizado(ruta):
    """
    Carga modelo personalizado con validación de archivos.
    """
    # Verificar existencia de archivos esenciales
    archivos_requeridos = ["config.json", "pytorch_model.bin"]
    for archivo in archivos_requeridos:
        if not os.path.exists(os.path.join(ruta, archivo)):
            raise FileNotFoundError(f"Archivo necesario no encontrado: {archivo}")
    
    tokenizador = AutoTokenizer.from_pretrained(ruta)
    modelo = AutoModelForCausalLM.from_pretrained(
        ruta,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    return modelo, tokenizador

def crear_interfaz_web(chatbot):
    """
    Crea interfaz de chat interactiva con gestión de contexto.
    """
    def responder(mensaje, historia):
        respuesta = chatbot.responder(mensaje)
        return historia + [(mensaje, respuesta)]
    
    return gr.ChatInterface(
        fn=responder,
        title="Chatbot IA Personalizado",
        description="Di hola para empezar la conversación",
        theme="soft",
        examples=["Explícame el machine learning", "Cuéntame un chiste técnico"],
        cache_examples=False
    )

def main_despliegue():
    # Paso 1: Cargar y preparar modelo base
    modelo_base, tokenizador_base = cargar_modelo("gpt2")
    
    # Paso 2: Personalizar y guardar el modelo
    modelo_peft = configurar_peft(modelo_base)
    guardar_modelo(modelo_peft, tokenizador_base, "./modelo_personalizado")
    
    # Paso 3: Cargar el modelo personalizado
    try:
        modelo, tokenizador = cargar_modelo_personalizado("./modelo_personalizado")
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        return
if __name__ == "__main__":
    main_despliegue()