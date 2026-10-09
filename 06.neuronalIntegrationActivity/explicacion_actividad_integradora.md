# Nuestra actividad: `actividad_integradora.ipynb`, explicada desde cero

> Idea en una frase: el mismo juego del ejemplo del profesor, pero con **cápsulas de fábrica** (buena vs. defectuosa) y con **tres escalones** en vez de dos: CNN propia → Transfer Learning → Fine-Tuning.

## 0. Qué cambia respecto del ejemplo (y por qué)

| | Ejemplo del profesor | Nuestra actividad |

|---|---|---|
| Fotos | Flores (5 clases) | Cápsulas MVTec-AD (2 clases: buena/defectuosa) |
| Origen de datos | Internet (`tfds`) | Disco local o Colab (`datasets/mvtec-ad/capsule/`) |
| Salida de la red | `Dense(5, softmax)` = reparte 100% entre 5 | `Dense(1, sigmoid)` = una sola probabilidad: "¿qué tan defectuosa es?" (más de 0.5 = defectuosa) |
| Error que se mide | `categorical_crossentropy` (varias clases) | `binary_crossentropy` (dos clases) |
| Métricas extra | Solo accuracy | + matriz de confusión y reporte (precisión/recall/F1) |
| Escalones | 2 (TL y hecha a mano) | 3 (CNN → TL → FT) |

## 1. La materia prima: el dataset (§1, celdas 2–4)

- **Celda de setup (Colab)**: si la carpeta de datos no existe, instala el descargador y baja el recorte `capsule` (~387 MB). Si ya existe, no hace nada. En tu PC se saltea.
- **Estructura**: `train/good/` (219 buenas) + `test/` con 6 carpetas (`good` + 5 tipos de defecto: crack, faulty_imprint, poke, scratch, squeeze). Total: 351 fotos de 1000×1000.
- **Truco importante**: el dataset original solo trae buenas para entrenar (está pensado para otra técnica). Como nosotros hacemos clasificación clásica, **juntamos todo y re-etiquetamos**: `good` → 0, cualquier defecto → 1. Quedan 242 buenas y 109 defectuosas.
- La celda 4 cuenta fotos por carpeta y te muestra una buena al lado de una rota: siempre mirá los datos con tus ojos antes de entrenar.

## 2. Preparar las fotos (§2, celda 6)

Tres pasos, igual que en el ejemplo pero hechos a mano porque los datos están en disco:

1. **Etiquetar**: recorrer carpetas y anotar 0 o 1 según el nombre de la carpeta.
2. **Achicar a 150×150**: mismo molde del ejemplo (menos píxeles = entrena muchísimo más rápido).
3. **Separar 80/20 con `stratify`**: 80% para entrenar, 20% test final. *Estratificado* = el 20% conserva la proporción buenas/defectuosas (si lo hicieras a ojo te podría tocar un test sin defectos y la prueba no valdría nada). Además el `fit` aparta otro 20% para validación.

## 3. Escalón 1: CNN desde cero (§3–§5, celdas 8–13)

Es el "bebé" del ejemplo: `Rescaling + 3 bloques Conv/Pool + Dense(50/20/1)`. Todo aleatorio al inicio, todo por aprender.

- **Celda 10**: `compile` + `fit` idénticos al ejemplo (adam, 50 epochs, batch 32, `EarlyStopping` con paciencia 5). Misma receta a propósito: así la comparación es justa.
- **Celda 11**: las curvas de accuracy/loss (se leen igual que en el ejemplo: manda la curva de validación).
- **Celda 13**: la evaluación con tres miradas:
  - `evaluate` → loss y accuracy en el test.
  - `predict > 0.5` → convertir probabilidades en decisiones ("más de 0.5 = defectuosa").
  - **Matriz de confusión** → tabla de 2×2: cuántas buenas acertó, cuántas defectuosas acertó y dónde se equivocó.
  - **Reporte** → `precision` (de las que marqué defectuosas, ¿cuántas lo eran?), `recall` (de las defectuosas reales, ¿cuántas atrapé?) y `F1` (promedio de ambas). En una fábrica el **recall** es sagrado: prefiero revisar de más antes que dejar pasar una cápsula rota.

## 4. Escalón 2: Transfer Learning (§6, celdas 15–18)

El "cocinero experto", igual que el camino A del ejemplo:

- **Celda 15**: se carga VGG16 congelado y se le pone el mismo cabezal del ejemplo. El `summary` muestra los ~14.7M de parámetros como *non-trainable*: es la prueba de que la base no aprende, solo la cabeza.
- **Celda 16**: las fotos se preparan con `preprocess_input` (el idioma de VGG16) en copias nuevas (`X_train_tl`), porque los arrays originales están en 0–255. Mismo `fit` que la CNN.
- **Celdas 17–18**: mismas curvas y métricas. Se guarda `y_pred_tl` (las decisiones del TL) porque después lo vamos a necesitar.

## 5. Escalón 3: Fine-Tuning (§7, celdas 20–21)

**La analogía**: el cocinero experto ya trabaja en tu restaurante (escalón 2). Ahora le permitís ajustar *apenas* sus últimas técnicas a tus platos, con indicaciones suaves.

- **Celda 20**: se libera **solo** el último bloque (`block5_conv*`, lo imprime para que lo verifiques), se **recompila** con `Adam(1e-5)` —un learning rate (paso de corrección) cien veces más chico que lo normal— y se entrena 15 epochs con paciencia 3.
  - ¿Por qué LR tan bajo? Porque si corregís fuerte, destruís los 20 años de experiencia del cocinero.
  - ¿Por qué recompilar? Porque cambiar `trainable` después de compilar no tiene efecto: es como cambiar las reglas a mitad del partido sin avisarle al árbitro.
- **Celda 21**: curvas + métricas del modelo ajustado.
- Detalle clave: el fine-tuning modifica `model_tl` en el lugar. Por eso en el escalón 2 guardamos `y_pred_tl`: sin ese array, el resultado del TL congelado se perdería.

## 6. La comparativa y el informe (§8–§9, celdas 23–24)

- **Celda 23**: una tablita con `acc / precision / recall / f1` de los tres, sobre el **mismo** test. Usa los arrays guardados (`y_pred`, `y_pred_tl`) más la predicción fresca del FT. Resultado esperable con tan pocos datos: `CNN < TL`, y `FT` igual o un poquito mejor que `TL`.
- **§9 + `informe.md`**: pegar ahí la tabla, las curvas y las matrices, y responder: ¿cuánto ganó el TL? ¿aportó el FT? ¿Qué errores quedan?

## 7. Auto-chequeo (tápate las respuestas)

1. ¿Por qué la salida es `Dense(1, sigmoid)` y el loss `binary_crossentropy` en vez de lo del ejemplo?
2. ¿Qué hace `stratify` y qué saldría mal sin él?
3. ¿Por qué se guarda `y_pred_tl` antes del fine-tuning?
4. Si el recall de "defectuosa" es bajo pero el accuracy es alto, ¿el modelo sirve para la fábrica? (Pista: pensá en las proporciones 242 vs 109.)
5. ¿Qué dos cosas pasarían si en el FT usaras learning rate normal y descongelaras todo?
