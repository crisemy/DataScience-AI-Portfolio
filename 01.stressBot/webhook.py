from flask import Flask, request, jsonify
from reglas_difusas import evaluador_estres

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    
    params = data['queryResult']['parameters']

    # Obtener los parámetros desde Dialogflow
    horas = float(params.get('horas_sueno', 0))
    carga = float(params.get('carga_laboral', 0))
    emocional = float(params.get('estado_emocional', 0))
    dolores = float(params.get('dolores_fisicos', 0))
    concentracion = float(params.get('concentracion', 0))

    # Asignar valores al sistema difuso
    evaluador_estres.input['horas_sueno'] = horas
    evaluador_estres.input['carga_laboral'] = carga
    evaluador_estres.input['estado_emocional'] = emocional
    evaluador_estres.input['dolores_fisicos'] = dolores
    evaluador_estres.input['concentracion'] = concentracion

    # Calcular el nivel de estrés
    evaluador_estres.compute()
    nivel = evaluador_estres.output['nivel_estres']

    # Agregar interpretación cualitativa
    if nivel < 3.5:
        evaluacion = "🙂 Estrés Bajo. Chill Marley"
    elif nivel < 6:
        evaluacion = "😐 Estrés Moderado. Tranquilo, no te vayas a estresar"
    elif nivel < 8.5:
        evaluacion = "😟 Estrés Alto. Necesitas salir a tomar aire libre"
    else:
        evaluacion = "😵‍💫 Estrés Crítico. Es hora de tomarte vacaciones"

    # Armar respuesta completa
    mensaje = (
        f"Según tus respuestas, tu nivel estimado de estrés es: {nivel:.2f} en una escala de 0 a 10.\n"
        f"{evaluacion}"
    )

    return jsonify({"fulfillmentText": mensaje})

if __name__ == '__main__':
    app.run(port=8080, debug=True)

