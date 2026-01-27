import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Variables de entrada
horas_sueno = ctrl.Antecedent(np.arange(0, 11, 1), 'horas_sueno')
carga_laboral = ctrl.Antecedent(np.arange(0, 11, 1), 'carga_laboral')
estado_emocional = ctrl.Antecedent(np.arange(0, 11, 1), 'estado_emocional')
dolores_fisicos = ctrl.Antecedent(np.arange(0, 8, 1), 'dolores_fisicos')
concentracion = ctrl.Antecedent(np.arange(0, 11, 1), 'concentracion')

# Variable de salida
nivel_estres = ctrl.Consequent(np.arange(0, 11, 1), 'nivel_estres')

# Funciones de membresía automáticas
horas_sueno.automf(3)         # poor, average, good
carga_laboral.automf(3)       # poor, average, good
estado_emocional.automf(3)    # poor, average, good
concentracion.automf(3)       # poor, average, good

# Funciones de membresía personalizadas
dolores_fisicos['nunca'] = fuzz.trimf(dolores_fisicos.universe, [0, 0, 2])
dolores_fisicos['ocasionalmente'] = fuzz.trimf(dolores_fisicos.universe, [1, 3, 5])
dolores_fisicos['frecuentemente'] = fuzz.trimf(dolores_fisicos.universe, [4, 6, 7])

nivel_estres['bajo'] = fuzz.trimf(nivel_estres.universe, [0, 0, 3])
nivel_estres['moderado'] = fuzz.trimf(nivel_estres.universe, [2, 5, 7])
nivel_estres['alto'] = fuzz.trimf(nivel_estres.universe, [6, 7.5, 9])
nivel_estres['critico'] = fuzz.trimf(nivel_estres.universe, [8, 10, 10])

# Reglas de producción (corrigiendo 'high' por 'good')
reglas = [
    ctrl.Rule(horas_sueno['poor'] & carga_laboral['good'], nivel_estres['critico']),
    ctrl.Rule(horas_sueno['average'] & estado_emocional['poor'], nivel_estres['critico']),
    ctrl.Rule(carga_laboral['average'] & dolores_fisicos['frecuentemente'], nivel_estres['alto']),
    ctrl.Rule(estado_emocional['good'] & horas_sueno['good'], nivel_estres['bajo']),
    ctrl.Rule(concentracion['poor'] & estado_emocional['average'], nivel_estres['alto']),
    ctrl.Rule(concentracion['good'] & estado_emocional['good'], nivel_estres['bajo']),
    ctrl.Rule(horas_sueno['good'] & carga_laboral['poor'], nivel_estres['bajo']),
    ctrl.Rule(estado_emocional['average'] & dolores_fisicos['ocasionalmente'], nivel_estres['moderado']),
    ctrl.Rule(horas_sueno['average'] & concentracion['average'], nivel_estres['moderado']),
    ctrl.Rule(carga_laboral['average'] & estado_emocional['average'], nivel_estres['moderado']),
]

# Sistema difuso
sistema_estres = ctrl.ControlSystem(reglas)
evaluador_estres = ctrl.ControlSystemSimulation(sistema_estres)
