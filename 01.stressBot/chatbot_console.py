from reglas_difusas import evaluador_estres

def chatbot():
    print("👋 Hola, soy tu Asistente Virtual para evaluación de estrés.")
    print("Contesta con valores entre 0 y 10 (o lo que se indique):")

    evaluador_estres.input['horas_sueno'] = float(input("😴 ¿Cuántas horas dormiste anoche? "))
    evaluador_estres.input['carga_laboral'] = float(input("📈 ¿Qué tan alta fue tu carga laboral? "))
    evaluador_estres.input['estado_emocional'] = float(input("💢 ¿Cómo describirías tu estado emocional? "))
    evaluador_estres.input['dolores_fisicos'] = float(input("🤕 ¿Cuántos días esta semana tuviste dolor físico? (0-7) "))
    evaluador_estres.input['concentracion'] = float(input("🧠 ¿Qué tan bien pudiste concentrarte hoy? "))

    evaluador_estres.compute()
    resultado = evaluador_estres.output['nivel_estres']
    
    print(f"\n🔍 Resultado: Tu nivel estimado de estrés es {resultado:.2f}/10")

    if resultado < 3.5:
        print("🙂 Estrés Bajo. Chill Marley")
    elif resultado < 6:
        print("😐 Estrés Moderado. Tranquilo, no te vayas a estresar")
    elif resultado < 8.5:
        print("😟 Estrés Alto. Necesitas salir a tomar aire libre")
    else:
        print("😵‍💫 Estrés Crítico. Es hora de tomarte vacaciones")
