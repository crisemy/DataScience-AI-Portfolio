# El ejemplo del profesor: `transfer_learning.ipynb`, explicado desde cero

> Idea en una frase: enseñarle a la compu a distinguir **5 tipos de flores**, de dos maneras: (A) reutilizando un "cerebro" ya entrenado y (B) creando uno desde cero. Después se comparan.

## 0. Vocabulario mínimo (sin esto no se entiende nada)

| Palabra | Qué es, en criollo |

|---|---|
| Red neuronal | Una receta con muchos "botones" (pesos) que se ajustan solos probando y corrigiendo. |
| Imagen para la compu | Una tabla gigante de números (cada píxel = 3 números: rojo, verde, azul). |
| Etiqueta (*label*) | La respuesta correcta de cada foto ("esto es una margarita"). |
| *Epoch* | Una vuelta completa: mostrarle **todas** las fotos una vez. |
| *Batch* | De a cuántas fotos aprende por vez (ej. 32). No mira todo junto, va por puñados. |
| *Accuracy* | % de aciertos. |
| *Loss* (pérdida) | Qué tan mal le va (número: más bajo = mejor). El entrenamiento intenta bajarlo. |
| Validación | Fotos que el modelo **no usa para aprender**, solo para controlar si está memorizando en vez de aprender (trampa típica = *overfitting*). |
| *EarlyStopping* | "Si en 5 vueltas seguidas no mejorás en validación, paramos y nos quedamos con tu mejor versión". Evita entrenar de más. |

## 1. Qué datos usa (celdas 1–7)

- **Dataset `tf_flowers`**: ~3.700 fotos de flores en 5 clases. Se piden con `tfds.load(...)` en dos partes: 70% para entrenar, 30% para test final.
- **`resize` a 150×150**: todas las fotos se achican al mismo tamaño, porque la red exige medidas fijas (como un molde: todo lo que entra tiene que tener la misma forma).
- **`to_categorical`**: convierte la etiqueta `"3"` en `[0,0,0,1,0]`. La red no entiende "margarita", entiende listas de números.

## 2. El camino A: Transfer Learning con VGG16 (celdas 8–17)

**La analogía**: VGG16 es un cocinero con 20 años de experiencia (entrenado con millones de fotos de ImageNet). No lo mandás a estudiar cocina desde cero: lo contratás y **solo le enseñás tu carta** (las 5 flores).

- **Celda 9**: se carga `VGG16(weights="imagenet", include_top=False)` y se congela (`trainable = False`).
  - `weights="imagenet"` = trae el cerebro ya entrenado.
  - `include_top=False` = le sacamos su "cabeza" original (que clasificaba 1000 cosas) porque vamos a ponerle la nuestra.
  - `trainable = False` = **congelar**: sus 14 millones de valores no se tocan, solo se usan. Aprender = solo la cabeza nueva.
  - `preprocess_input` = prepara las fotos como VGG16 las espera (es su "idioma"). Por eso acá **no** se usa `Rescaling`.
- **Celda 12**: se le agrega la cabeza nueva: `Flatten` (aplanar la info) + `Dense(50)` + `Dense(20)` + `Dense(5, softmax)`. `softmax` = "repartí 100% de probabilidad entre las 5 flores".
- **Celda 15**: `compile(adam, categorical_crossentropy)` = elegir el método de estudio (`adam` = cómo corrige los botones; `categorical_crossentropy` = la regla para medir el error cuando hay varias clases). Luego `fit(..., validation_split=0.2, EarlyStopping(patience=5))`: entrena hasta 50 vueltas, usando 20% de los datos solo para controlar, y frena si no mejora.
- **Celda 16**: los gráficos. Cómo leerlos: la curva de *training* casi siempre mejora; la que importa es la de *validación*. Si training sube y validación se estanca o baja → está **memorizando** (overfitting).
- **Celda 17**: `evaluate` con las fotos de test (las que nunca vio). Resultado guardado: **accuracy ≈ 0.99**. Impresionante, pero lógico: el cerebro ya sabía ver.

## 3. El camino B: modelo hecho a mano (celdas 18–21)

**La analogía**: ahora en vez del cocinero experto, ponemos a un bebé a aprender cocina desde cero, mirando solo nuestras fotos de flores.

- **Celda 19**: `Sequential` con `Rescaling(1/255)` (convierte píxeles 0–255 a 0–1, números más cómodos) + 3 bloques `Conv2D + MaxPooling` (los "ojos" que detectan bordes y formas, cada vez más complejos) + `Flatten + Dense(50/20/5)`. Todo parte de valores al azar: **todo** tiene que aprender.
- Mismo `compile` y `fit` que el camino A, para comparar en igualdad de condiciones.
- Resultado guardado: **accuracy ≈ 0.82**. Bien, pero 17 puntos abajo del camino A.

## 4. La moraleja (lo que el profesor quiere que veas)

| | Camino A (Transfer Learning) | Camino B (desde cero) |

|---|---|---|
| Punto de partida | Cerebro experto (ImageNet) | Cerebro bebé (azar) |
| Qué aprende | Solo la cabeza nueva | Todo |
| Dato necesario | Poco | Mucho |
| Test | ~0.99 | ~0.82 |

**Conclusión**: con pocos datos, reutilizar conocimiento gana por lejos. El camino B necesitaría muchísimas más fotos para alcanzar al A.

## 5. Auto-chequeo (tápate las respuestas)

1. ¿Qué significa congelar (`trainable = False`) y por qué se hace *antes* del `compile`?
2. ¿Por qué el camino A usa `preprocess_input` y el B usa `Rescaling`? (Pista: son excluyentes.)
3. ¿Qué mirás en los gráficos para detectar overfitting?
4. ¿Por qué ambos modelos usan el mismo `fit` (epochs, batch, patience)?
