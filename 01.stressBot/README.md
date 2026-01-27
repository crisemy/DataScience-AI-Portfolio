# StressBot: Sistema Experto Difuso para Evaluación de Estrés

Este proyecto es parte de una entrega académica para la Maestría en Ciencia de Datos. Se trata de un **Sistema Experto Difuso (SED)** que simula un asistente virtual capaz de evaluar el nivel de estrés de una persona en función de cinco variables clave.

---

## Objetivo

Demostrar el funcionamiento de un sistema experto basado en **lógica difusa**, aplicando reglas de producción y un motor de inferencia para realizar una estimación inteligente del nivel de estrés, tanto desde consola como desde un chatbot en la nube.

---

## Tecnologías utilizadas

- Python 3.x
- [scikit-fuzzy (`skfuzzy`)] (https://github.com/scikit-fuzzy/scikit-fuzzy)
- [Flask] (https://flask.palletsprojects.com/en/stable/) (para exponer el sistema como webhook)
- [ngrok] (https://ngrok.com/) (para exponer el webhook a internet)
- [DialogFlow] (https://dialogflow.cloud.google.com/) (agente conversacional)
- JSON para estructura de datos en el webhook

---

## Variables lingüísticas difusas

El sistema toma como entrada cinco variables subjetivas, modeladas como **antecedentes difusos**:

- `horas_sueno` (0–10): Cantidad de horas dormidas
- `carga_laboral` (0–10): Nivel de exigencia laboral
- `estado_emocional` (0–10): Estabilidad emocional subjetiva
- `dolores_fisicos` (0–7): Días con dolor físico durante la semana
- `concentracion` (0–10): Nivel de enfoque mental

**Salida del sistema:**
- `nivel_estres` (0–10): Nivel de estrés estimado

---

## Motor de inferencia y reglas

Se utilizaron 10 reglas de producción difusas, por ejemplo:

- SI horas de sueño ES pobres Y carga laboral ES alta ENTONCES estrés ES crítico
- SI concentración ES pobre Y estado emocional ES regular ENTONCES estrés ES alto
- SI estado emocional ES bueno Y dolores físicos SON bajos ENTONCES estrés ES bajo

Estas reglas están implementadas usando el sistema Mamdani a través de la biblioteca `skfuzzy`.

---

## Ejecución del sistema

### 1. Ejecutar el sistema experto en consola

```bash
python main.py
```

Este script solicita al usuario ingresar los valores uno por uno desde la consola y devuelve el nivel de estrés estimado, junto con una interpretación textual.

---

### 2. Ejecutar como webhook para integración con DialogFlow

```bash
python webhook.py
```

Este script lanza un servidor Flask escuchando en `localhost:8080/webhook`.

Para integrarlo con DialogFlow:

- Configurar un intent con los parámetros: `horas_sueno`, `carga_laboral`, `estado_emocional`, `dolores_fisicos`, `concentracion`.
- Enviar los datos al endpoint `https://<tu-url-ngrok>/webhook`.

---

## Cómo exponer tu webhook usando ngrok

Para que DialogFlow acceda a tu servidor Flask local:

1. Instalar `ngrok` desde [https://ngrok.com](https://ngrok.com)
2. Ejecutar:

```bash
ngrok http 8080
```

3. Copiar la URL generada (`https://xxxx.ngrok.io`) y configurarla como webhook en DialogFlow.

---

## Pruebas y entrenamiento

El intent principal en DialogFlow fue entrenado con frases como:

- "Dormí 5 horas, dolor 4, emocional 5"
- "Hoy dormí 2 horas, me duele mucho el cuerpo y estoy mal"
- "No dormí nada, concentración 3, carga laboral 9"
- "Me siento bien emocionalmente pero me duele la cabeza"
- "Tengo mucho trabajo, dormí poco y no me puedo concentrar"

Estos ejemplos permiten extraer múltiples parámetros de una sola frase mediante NLP, para alimentar el sistema experto automáticamente.

---

## Archivos principales

- `main.py`: interfaz en consola del sistema difuso.
- `reglas_difusas.py`: contiene las definiciones del sistema experto, universos de discurso, reglas y lógica difusa.
- `chatbot_console.py`: contiene las respuestas que son enviadas al usuario.
- `webhook.py`: expone el sistema como un webhook para consumir desde DialogFlow.
- `README.md`: este archivo.
---