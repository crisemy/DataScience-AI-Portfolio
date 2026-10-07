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

> Pendiente de ejecución del notebook (requiere `requirements.txt` + descarga de pesos ImageNet, ~528 MB). Pegar aquí la tabla de §8:

| Enfoque | acc | precision | recall | f1 |

|---|---|---|---|---|
| CNN desde cero | — | — | — | — |
| TL congelado | — | — | — | — |
| FT bloque5 | — | — | — | — |

## 6. Conclusiones

> A completar tras la ejecución, respondiendo: ¿cuánto ganó el TL sobre la CNN con tan pocos datos? ¿aportó algo el FT? ¿Qué errores quedan (ver matrices de confusión de §5–§7)?
