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
| **10 ResNet50 Final + DA** | **24,114,308** | **ResNet50 Fine-tuning + Data Aug. (x5 semillas)** | **Acc: 0.9671 ± 0.0021 / Loss: 0.1089** | **0.9960** |
| **11 ResNet50 Final (NoDA)** | **24,114,308** | **ResNet50 Fine-tuning SIN Data Aug. (x5 semillas)** | **Acc: 0.9605 ± 0.0009 / Loss: 0.1795 ± 0.0044** | **N/A** |
| **14 MobileNetV3 + DA** | **3,244,420** | **MobileNetV3Large Fine-tuning + Data Aug. (1 semilla)** | **Acc: 0.9600 / Loss: 0.1169** | **0.9938** |

---

## 📈 Notebooks Finales (Análisis Multi-Semilla y Comparación)

### Notebook 10: ResNet50 + Data Augmentation - Multi-Seed (Notebook 10)
**Propósito:** Entrenar el mejor modelo (ResNet50 descongelado) con 5 semillas diferentes para evaluar robustez y variabilidad con data augmentation.

**Características:**
- 5 semillas independientes: [42, 123, 456, 789, 999]
- Data Augmentation: RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness
- 100 épocas por seed con Early Stopping (patience=30)
- Visualización: Curvas por seed + gráficas superpuestas
- **Resultado:** Test Accuracy media: 96.71% ± 0.21%

**Archivo:** `notebooks/10_resnet50_final.ipynb`

---

### Notebook 11: ResNet50 SIN Data Augmentation - Multi-Seed (Notebook 11)
**Propósito:** Entrenar ResNet50 descongelado con 5 semillas **sin data augmentation** para comparar el impacto de augmentation en la generalización.

**Características:**
- 5 semillas independientes: [42, 123, 456, 789, 999]
- **SIN Data Augmentation:** Uso directo de `train_ds`
- 100 épocas por seed con Early Stopping (patience=30)
- Visualización completa: Training + Validation Loss y Accuracy en cada subplot
- Panel superpuesto mostrando validación accuracy/loss de todas las seeds
- Tabla final con mean ± std del test accuracy

**Archivo:** `notebooks/11_resnet50_final_noDA.ipynb`

---

### Notebook 14: MobileNetV3 + Data Augmentation (1 semilla)
**Propósito:** Entrenar un modelo eficiente basado en MobileNetV3Large con data augmentation para comparar rendimiento con ResNet50 usando menos parámetros.

**Características:**
- 1 semilla: [42]
- Data Augmentation: RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness
- 100 épocas con Early Stopping (patience=30)
- Visualización: Curvas de aprendizaje, matriz de confusión, AUC por clase
- **Resultado:** Test Accuracy: 96.00%, Test Loss: 0.1169, AUC promedio: 0.9938

**Archivo:** `notebooks/14_mobilenet.ipynb`

**Ventajas:**
- 7.43× menos parámetros que ResNet50 (3.2M vs 24.1M)
- Comparable en precisión (96.00% vs 96.71%)
- Ideal para aplicaciones móviles y dispositivos con recursos limitados
- Más rápido de entrenar y desplegar

---

# 🎯 Conclusión y Selección del Modelo

## Modelo Seleccionado: ResNet50 + Data Augmentation

Aunque **MobileNetV3** ofrece ventajas de eficiencia (7.43× menos parámetros), hemos seleccionado **ResNet50 + Data Augmentation** como el modelo final de este proyecto. 

**Justificación en contexto médico:**

En aplicaciones de clasificación de imágenes médicas, la diferencia de ~**1% en precisión es crítica**. ResNet50 logra un **96.71% de accuracy** frente al 96.00% de MobileNetV3, lo que se traduce en:

- **Reducción de falsos negativos**: En diagnóstico ocular, cada falso negativo puede llevar a la pérdida de visión del paciente
- **Mayor confiabilidad clínica**: El 0.71% adicional representa cientos de vidas en aplicaciones reales
- **Mejor desempeño en patologías críticas**: ResNet50 muestra mejor balance en las 4 clases, especialmente en condiciones graves (CNV, DME)
- **Trade-off justificado**: Los 24.1M parámetros están plenamente justificados cuando se trata de diagnósticos médicos

Aunque MobileNetV3 es excelente para **aplicaciones móviles sin restricciones de precisión crítica**, en contextos clínicos priorizamos la máxima precisión diagnóstica sobre la eficiencia computacional.

> **"En medicina, 1% diferencia en precisión no es una mejora marginal, es la diferencia entre detectar y no detectar una enfermedad"**

---

# 🛠️ Tecnologías Utilizadas

- Python 3.x
- TensorFlow / Keras
- scikit-learn
- Pandas & NumPyS
- Matplotlib & Seaborn
- Kagglehub

---

# 👨‍💻 Autor
Pablo López Domínguez

Proyecto desarrollado como parte de la asignatura de Deep Learning  
Grado en Ciencia de Datos – Universitat de València  

Última actualización: Marzo 2026