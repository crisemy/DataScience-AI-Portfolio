# Clasificación automática de defectos en productos industriales mediante Redes Neuronales Convolucionales

## 1. Problema abordado

Desarrollar un modelo de Deep Learning capaz de clasificar imágenes de productos industriales según su estado visual. Se plantea como clasificación **binaria**: pieza `buena (0)` vs. pieza `defectuosa (1)`, caso típico de inspección visual automatizada donde un falso negativo (dejar pasar un defecto) cuesta más que un falso positivo.

## 2. Dataset empleado

MVTec-AD, categoría `capsule` ([proyecto original](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), licencia CC BY-NC-SA 4.0). Recorte local en `datasets/mvtec-ad/capsule/` (ver `README.md` para la descarga):

| Split | Clase | Imágenes |

|---|---|---|
| train | good | 219 |
| test | good | 23 |
| test | crack | 23 |
| test | faulty_imprint | 22 |
| test | poke | 21 |
| test | scratch | 23 |
| test | squeeze | 20 |

Total etiquetado: 351 imágenes (242 buenas / 109 defectuosas, 1000×1000 RGB). Las máscaras de `ground_truth/` no se usan en clasificación. Preprocesamiento: `resize` a 150×150 y split estratificado 80/20 train/test; el `fit` reserva además 20% de validación.

## 3. Modelo y proceso de entrenamiento

Tres enfoques encadenados (`actividad_integradora.ipynb`):

1. **CNN desde cero** (baseline): `Rescaling + Conv16(k10)/Pool3 + Conv32(k8)/Pool2 + Conv32(k6)/Pool2 + Flatten + Dense(50/20/1-sigmoid)`, pesos aleatorios.
2. **Transfer Learning**: `VGG16(imagenet, include_top=False)` congelado (`trainable=False` antes de compilar) + mismo cabezal denso; solo el cabezal aprende. Entrada con `preprocess_input`, no `Rescaling`.
3. **Fine-Tuning**: se libera solo el bloque final (`block5_conv*`), se recompila con `Adam(1e-5)` y se re-entrena.

| Hiperparámetro | CNN | TL | FT |

|---|---|---|---|
| Loss / optimizer | binary_crossentropy / adam | idem | binary_crossentropy / Adam(1e-5) |
| Epochs (máx.) | 50 | 50 | 15 |
| Batch size | 32 | 32 | 32 |
| EarlyStopping (`val_accuracy`) | patience=5 | patience=5 | patience=3 |

## 4. Métricas de performance

Además de `accuracy/loss` y sus curvas se reportan matriz de confusión y `precision/recall/F1` sobre el mismo split de test: en inspección industrial el `recall` de la clase defectuosa es la métrica crítica (minimizar defectos no detectados).

## 5. Resultados obtenidos

Resultados sobre el mismo split de test estratificado (71 imágenes: 49 buenas / 22 defectuosas), extraídos de §8 del notebook (`actividad_integradora.ipynb`, celdas 13/18/21/23). Precisión/recall/F1 corresponden a la clase positiva `defectuosa (1)` (average binario):

| Enfoque | loss (test) | acc | precision | recall | f1 |

|---|---|---|---|---|---|
| CNN desde cero | 0.6216 | 0.6901 | 0.0000 | 0.0000 | 0.0000 |
| TL congelado (VGG16) | 0.4113 | 0.8028 | 0.7222 | 0.5909 | 0.6500 |
| FT bloque5 (`Adam(1e-5)`) | 0.3585 | 0.8310 | 0.7273 | 0.7273 | 0.7273 |

Matrices de confusión (filas = real, columnas = predicho; orden `[good, defectuosa]`):

| Enfoque | TN (good→good) | FP (good→def.) | FN (def.→good) | TP (def.→def.) |

|---|---|---|---|---|
| CNN desde cero | 49 | 0 | 22 | 0 |
| TL congelado | 44 | 5 | 9 | 13 |
| FT bloque5 | 43 | 6 | 6 | 16 |

Lectura:

- **CNN:** colapsa a la clase mayoritaria — predice todo `good`. El accuracy (0.69) replica la proporción de buenas en test (49/71 ≈ 0.69) y es inútil para inspección: recall 0% de defectos. El entrenamiento lo anticipa: `val_accuracy` queda clavada en 0.8036 desde la epoch 1 y `EarlyStopping` corta en la epoch 6.
- **TL:** el salto cualitativo. Con solo el cabezal entrenado (410.691 params. entrenables vs. 14.714.688 congelados) pasa a F1 = 0.65 en defectos; el mejor `val_accuracy` (0.8750, epoch 4) corta en epoch 9. Quedan 9 falsos negativos: ese es el costo operativo crítico.
- **FT:** liberar solo `block5_conv1/2/3` con LR 1e-5 baja el loss de 0.41 a 0.36 y sube el recall de 0.5909 a 0.7273 (+13.6 pp, 3 defectos más atrapados: FN 9→6) a costa de 1 falso positivo extra (5→6). F1 0.65→0.73.

## 6. Conclusiones

1. **¿Cuánto ganó el TL sobre la CNN con tan pocos datos?** Todo: de un clasificador constante inútil (F1 = 0) a F1 = 0.65 y +11 pp de accuracy, sin agregar datos. Con 280 imágenes de entrenamiento, reutilizar los filtros de ImageNet (bordes, texturas, contrastes) gana por lejos a aprenderlos desde cero — igual que en el ejemplo del profesor (flores: 0.82→0.99).
2. **¿Aportó algo el FT?** Sí, y donde más importa: el recall de defectos sube de 0.59 a 0.73. El ajuste fino con LR bajo especializa los filtros de alto nivel a la geometría de la cápsula sin destruir el conocimiento previo (sin "amnesia catastrófica"). La mejora en accuracy (+2.8 pp) es modesta; la mejora en sensibilidad es la que justifica el FT.
3. **¿Qué errores quedan?** El mejor modelo (FT) todavía deja pasar 6 de 22 defectos (27%) y descarta 6 de 49 buenas (12%). Para una línea farmacéutica, un recall de 0.73 sigue siendo insuficiente como único filtro: serviría como pre-screening con revisión humana de los positivos, o requeriría más datos, aumento de datos, umbral < 0.5 (priorizar recall), o ponderación de clases. Métrica rectora a futuro: recall/F1 de `defectuosa`, no accuracy global.
