# 🔬 Clasificación de Imágenes Retinales OCT con Deep Learning

Proyecto de clasificación multiclase de imágenes médicas utilizando el dataset de Tomografía de Coherencia Óptica (OCT) retinal.


---

## 📊 Dataset

- **Fuente:** Kermany et al. (2018) - Mendeley Data V3  
  https://data.mendeley.com/datasets/rscbjbr9sj/3
- **Citación:**  
  Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018),  
  "Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images",  
  Mendeley Data, V3, doi: 10.17632/rscbjbr9sj.3
- **Total de imágenes:** 109,309 imágenes JPEG
- **Clases (4):**
  - CNV – Neovascularización Coroidea
  - DME – Edema Macular Diabético
  - DRUSEN
  - NORMAL
- **División realizada:**
  - Train: 70%
  - Validation: 20%
  - Test: 10%

### 📸 Ejemplos del Dataset

A continuación se muestran ejemplos de imágenes OCT retinales de cada una de las 4 clases:

![Ejemplos de Imágenes OCT por Categoría](output.png)

*Muestras representativas de las 4 condiciones retinales: CNV, DME, DRUSEN y NORMAL.*

---

## 🎯 Objetivo

Construir y evaluar modelos de Deep Learning capaces de clasificar automáticamente imágenes OCT en cuatro patologías retinales.

Problema de clasificación multiclase:

f(x) → y ∈ {CNV, DME, DRUSEN, NORMAL}

En contexto médico se prioriza especialmente la **sensibilidad (recall)** para minimizar falsos negativos.

---

# 🚀 Estado del Arte (SOTA)

A continuación se resumen estudios relevantes en clasificación OCT mediante Deep Learning:

| Modelo | Dataset | Accuracy | Otras Métricas | Referencia |
|--------|----------|----------|----------------|------------|
| OCTDeepNet2 (CNN personalizado) | OCT (4 clases) | 98% | Precision 0.98, Recall 1.00, F-Score 0.99 | Rajan & Kumar (2025) |
| Modified ResNet-50 + RF (EOCT) | OCT retinal | ≈97.88%–98.47% | Sensitivity 98.36%, Specificity 96.15%, Precision 97.40% | EOCT Study |
| Attention-Based DenseNet | OCT (4 clases) | 91.67% | — | ElShafie et al. (2025) |
| Hybrid ResNet50 + EfficientNetB0 | ~84,000 imágenes OCT | 97.50% | — | Hybrid Model (Springer) |
| Transfer Learning VGG16 modificado | OCT (70/20/10 split) | 97% | — | Transfer Learning Study |

---

## 🔗 Enlaces a los artículos SOTA

- **OCTDeepNet2 – Rajan & Kumar (2025)**  
  https://link.springer.com/article/10.1007/s42979-025-03715-w  

- **Modified ResNet-50 + RF (EOCT)**  
  https://www.mdpi.com/1424-8220/23/12/5393  

- **Attention-Based DenseNet – ElShafie et al. (2025)**  
  https://link.springer.com/article/10.1007/s00521-024-10450-5  

- **Hybrid ResNet50 + EfficientNetB0 (Springer)**  
  https://link.springer.com/article/10.1007/s11042-024-19922-1  

- **Transfer Learning VGG16 modificado**  
  https://link.springer.com/article/10.1007/s12596-025-02561-6  

---

# 📊 Métricas de Evaluación

Métricas principales:
- Accuracy
- Precision por clase
- Recall por clase
- F1-Score
- Macro F1-Score
- Matriz de Confusión

Métricas complementarias:
- AUC-ROC (One-vs-Rest)
- Cohen's Kappa
- Top-k Accuracy

En aplicaciones médicas es especialmente relevante:
- Alta sensibilidad (recall)
- Buena especificidad
- Control de falsos negativos

---

# 📊 Resultados de Modelos

## Modelos Implementados

| Modelo (Notebook) | Parámetros | Detalles / Configuración | Acc / Loss Test | AUC Promedio |
|-------------------|------------|--------------------------|-----------------|--------------|
| Modelo Lineal (Reg. Logística) | 49,156 | Imágenes 64×64 escala grises | 0.6021 / - | - |
| Árbol de Decisión | 875 nodos | Profundidad: 20 | 0.6027 / - | - |
| 1 Red Neuronal Simple | 131,236 | 1 capa oculta (32 neuronas) | 0.4700 / - | - |
| 2 CNN Básica | 4,287,620 | 3 capas Conv + 1 Densa (128x128) | 0.9121 / - | - |
| 3 Transfer Learning (VGG16) | 14,847,044 | VGG16 (ImageNet) Fine-tuning Total | Acc: 0.9640 / Loss: 0.1156 | 0.9938 |
| 5 Resnet50 | 24,114,308 | ResNet50 Transfer Learning base | Acc: 0.9341 / Loss: 0.2030 | 0.9842 |
| 6 Resnet50 Unfrozen | 24,114,308 | ResNet50 Fine-tuning Total | Acc: 0.9748 / Loss: 0.0961 | 0.9970 |
| 7 Resnet50 Unfrozen DA | 24,114,308 | ResNet50 Fine-tuning + Data Aug. | Acc: 0.9654 / Loss: 0.1055 | 0.9955 |
| 8 CNN DA | 4,287,620 | CNN Básica + Data Augmentation | *Pendiente* | *Pendiente* |
| 9 Swin Pretrained | 27,720,318 | Swin Transf. Tiny (Transfer L. + DA) | Acc: 0.9665 / Loss: 0.0986 | 0.9967 |



---

# 🛠️ Tecnologías Utilizadas

- Python 3.x
- TensorFlow / Keras
- scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Kagglehub

---

# 👨‍💻 Autor
Pablo López Domínguez

Proyecto desarrollado como parte de la asignatura de Deep Learning  
Grado en Ciencia de Datos – Universitat de València  

Última actualización: Febrero 2026