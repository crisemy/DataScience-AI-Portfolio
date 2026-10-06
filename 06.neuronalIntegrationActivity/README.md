# 06 — Actividad integradora: visión computacional

Clasificación de defectos en productos industriales con CNN sobre MVTec-AD (categoría `capsule`).

## Dataset: qué y dónde

* **Fuente:** mirror `foersben/mvtec-ad` en Hugging Face (espejo del [MVTec AD original](https://www.mvtec.com/company/research/datasets/mvtec-ad), licencia CC BY-NC-SA 4.0, solo uso no comercial).
* **Recorte usado:** solo la categoría `capsule` (~387 MB, 462 archivos) en vez del dataset completo (~7 GB).
* **Ubicación local (no versionada en git):**
  `06.neuronalIntegrationActivity/datasets/mvtec-ad/capsule/`
  * `train/good/` — 219 imágenes buenas (entrenamiento).
  * `test/{crack, faulty_imprint, good, poke, scratch, squeeze}/` — 132 imágenes.
  * `ground_truth/{crack, faulty_imprint, poke, scratch, squeeze}/` — 109 máscaras + `license.txt`, `readme.txt`.

## Cómo descargar (reproducible)

Requiere la CLI `hf` (`pip install -U huggingface_hub`, y `brotli>=1.2.0` — con `1.0.9` la descarga falla con `TypeError: process() takes no keyword arguments`).

```bash
# capsule (~387 MB) — la usada en la prueba
hf download foersben/mvtec-ad --repo-type dataset \
  --include "capsule/*" \
  --local-dir 06.neuronalIntegrationActivity/datasets/mvtec-ad

# otras categorías opcionales (tornillos / píldoras / frascos):
hf download foersben/mvtec-ad --repo-type dataset \
  --include "screw/*" \
  --local-dir 06.neuronalIntegrationActivity/datasets/mvtec-ad   # ~196 MB

hf download foersben/mvtec-ad --repo-type dataset \
  --include "pill/*" \
  --local-dir 06.neuronalIntegrationActivity/datasets/mvtec-ad    # ~275 MB

hf download foersben/mvtec-ad --repo-type dataset \
  --include "bottle/*" \
  --local-dir 06.neuronalIntegrationActivity/datasets/mvtec-ad  # ~157 MB
```

> Nota: el comando genera una carpeta temporal `.cache/huggingface/` dentro de `--local-dir`; se puede borrar tras la descarga. La carpeta `datasets/` no se commitea (ver `.gitignore`).
