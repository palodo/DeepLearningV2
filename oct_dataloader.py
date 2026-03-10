"""
DataLoader para imágenes OCT (Tomografía de Coherencia Óptica)
Módulo para cargar datasets de entrenamiento, validación y prueba
con configuración flexible de batch size, tamaño de imagen y subset de datos.
"""

import os
import tensorflow as tf
from typing import Tuple, Optional


def create_oct_dataloaders(
    data_path: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    color_mode: str = 'grayscale',
    train_subset_fraction: float = 1.0,
    seed: int = 42,
    validation_split: Optional[float] = None,
    optimize: bool = True,
    verbose: bool = True
):
    """
    Crea dataloaders para el dataset OCT con configuración flexible.
    
    Parámetros
    ----------
    data_path : str
        Ruta al directorio que contiene las carpetas 'train', 'val' y 'test'
    img_size : tuple, opcional
        Tamaño al que se redimensionarán las imágenes (ancho, alto)
        Por defecto: (224, 224)
    batch_size : int, opcional
        Número de imágenes por lote
        Por defecto: 32
    train_subset_fraction : float, opcional
        Fracción del dataset de entrenamiento a utilizar (entre 0.0 y 1.0)
        Por defecto: 1.0 (usar todo el dataset)
        Ejemplo: 0.5 para usar solo el 50% de los datos de entrenamiento
    seed : int, opcional
        Semilla para reproducibilidad
        Por defecto: 42
    validation_split : float, opcional
        Si se especifica, divide el conjunto de entrenamiento en train/val
        Útil si no existe una carpeta 'val' separada
        Por defecto: None
    optimize : bool, opcional
        Si True, aplica optimizaciones de rendimiento (cache + prefetch)
        Por defecto: True
    verbose : bool, opcional
        Si True, imprime información sobre los datasets creados
        Por defecto: True
    
    Retorna
    -------
    tuple
        (train_dataset, val_dataset, test_dataset, class_names)
        - train_dataset: tf.data.Dataset para entrenamiento
        - val_dataset: tf.data.Dataset para validación
        - test_dataset: tf.data.Dataset para prueba
        - class_names: lista con los nombres de las clases
    
    Ejemplo
    -------
    >>> train_ds, val_ds, test_ds, classes = create_oct_dataloaders(
    ...     data_path='./OCT2017',
    ...     img_size=(256, 256),
    ...     batch_size=64,
    ...     train_subset_fraction=0.5  # Usar solo el 50% para entrenar
    ... )
    """
    
    # Mapeo de categorías
    class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    
    if verbose:
        print("⚙️ Configuración de DataLoaders")
        print(f"   • Tamaño de imagen: {img_size}")
        print(f"   • Batch size: {batch_size}")
        print(f"   • Clases: {class_names}")
        print(f"   • Train subset: {train_subset_fraction*100:.1f}%")
        print(f"   • Seed: {seed}\n")
    
    # ========================================================================
    # DATA LOADER DE ENTRENAMIENTO
    # ========================================================================
    train_path = os.path.join(data_path, 'train')
    
    if not os.path.exists(train_path):
        raise ValueError(f"No se encontró la carpeta de entrenamiento en: {train_path}")
    
    if verbose:
        print("📦 Creando data loader de entrenamiento...")
    
    # Crear dataset de entrenamiento completo
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_path,
        labels='inferred',
        label_mode='int',
        class_names=class_names,
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=seed
    )
    
    # Aplicar subset si se especificó
    if train_subset_fraction < 1.0:
        # Calcular número de batches a usar
        total_batches = tf.data.experimental.cardinality(train_dataset).numpy()
        subset_batches = max(1, int(total_batches * train_subset_fraction))
        train_dataset = train_dataset.take(subset_batches)
        
        if verbose:
            print(f"   ⚠️  Usando {train_subset_fraction*100:.1f}% del dataset de entrenamiento")
            print(f"   ⚠️  Batches: {subset_batches} de {total_batches}")
    
    # Aplicar optimización si se especificó
    if optimize:
        train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        if verbose:
            print(f"✅ Data loader de entrenamiento creado (con optimización)")
    else:
        if verbose:
            print(f"✅ Data loader de entrenamiento creado")
    
    # ========================================================================
    # DATA LOADER DE VALIDACIÓN
    # ========================================================================
    val_path = os.path.join(data_path, 'val')
    val_dataset = None
    
    if os.path.exists(val_path):
        if verbose:
            print("\n📦 Creando data loader de validación...")
        
        val_dataset = tf.keras.utils.image_dataset_from_directory(
            val_path,
            labels='inferred',
            label_mode='int',
            class_names=class_names,
            color_mode=color_mode,
            batch_size=batch_size,
            image_size=img_size,
            shuffle=False
        )
        
        # Aplicar optimización
        if optimize:
            val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            if verbose:
                print(f"✅ Data loader de validación creado (con optimización)")
        else:
            if verbose:
                print(f"✅ Data loader de validación creado")
    
    elif validation_split is not None:
        if verbose:
            print("\n⚠️  No se encontró carpeta 'val', usando validation_split del train")
        # Aquí podrías implementar un split manual si lo necesitas
        pass
    else:
        if verbose:
            print("\n⚠️  No se encontró data loader de validación")
    
    # ========================================================================
    # DATA LOADER DE PRUEBA
    # ========================================================================
    test_path = os.path.join(data_path, 'test')
    test_dataset = None
    
    if os.path.exists(test_path):
        if verbose:
            print("\n📦 Creando data loader de prueba...")
        
        test_dataset = tf.keras.utils.image_dataset_from_directory(
            test_path,
            labels='inferred',
            label_mode='int',
            class_names=class_names,
            color_mode=color_mode,
            batch_size=batch_size,
            image_size=img_size,
            shuffle=False
        )
        
        # Aplicar optimización
        if optimize:
            test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            if verbose:
                print(f"✅ Data loader de prueba creado (con optimización)")
        else:
            if verbose:
                print(f"✅ Data loader de prueba creado")
    else:
        if verbose:
            print("\n⚠️  No se encontró data loader de prueba")
    
    # ========================================================================
    # INFORMACIÓN ADICIONAL
    # ========================================================================
    if verbose:
        print("\n" + "="*60)
        print("📊 RESUMEN DE DATASETS")
        print("="*60)
        
        if train_dataset:
            train_cardinality = tf.data.experimental.cardinality(train_dataset).numpy()
            print(f"Train:      {train_cardinality} batches")
        
        if val_dataset:
            val_cardinality = tf.data.experimental.cardinality(val_dataset).numpy()
            print(f"Validation: {val_cardinality} batches")
        
        if test_dataset:
            test_cardinality = tf.data.experimental.cardinality(test_dataset).numpy()
            print(f"Test:       {test_cardinality} batches")
        
        print("="*60 + "\n")
    
    return train_dataset, val_dataset, test_dataset, class_names


def get_label_map():
    """
    Retorna el mapeo de nombres de clase a índices numéricos.
    
    Retorna
    -------
    dict
        Diccionario con mapeo clase -> índice
    """
    return {
        'CNV': 0,      # Neovascularización Coroidea
        'DME': 1,      # Edema Macular Diabético
        'DRUSEN': 2,   # Drusas
        'NORMAL': 3    # Retina Normal
    }


def dataloader_to_arrays(dataset, dataset_name="Dataset", limit=None):
    """
    Convierte un tf.data.Dataset a arrays numpy (X, y).
    Útil para modelos de sklearn que requieren arrays en memoria.
    
    Parámetros
    ----------
    dataset : tf.data.Dataset
        Dataset de TensorFlow a convertir
    dataset_name : str, opcional
        Nombre del dataset para mensajes informativos
        Por defecto: "Dataset"
    limit : int, opcional
        Número máximo de muestras a extraer
        Si es None, extrae todas las muestras
        Por defecto: None
    
    Retorna
    -------
    tuple
        (X, y) donde:
        - X: np.ndarray con shape (n_samples, img_height, img_width, channels)
        - y: np.ndarray con shape (n_samples,) con las etiquetas
    
    Ejemplo
    -------
    >>> X_train, y_train = dataloader_to_arrays(train_dataset, "Train", limit=5000)
    >>> print(X_train.shape)  # (5000, 224, 224, 3)
    """
    import numpy as np
    
    X_list = []
    y_list = []
    total_samples = 0
    
    print(f"🔄 Convirtiendo {dataset_name} a arrays numpy...")
    
    for images, labels in dataset:
        X_list.append(images.numpy())
        y_list.append(labels.numpy())
        total_samples += images.shape[0]
        
        # Aplicar límite si se especificó
        if limit is not None and total_samples >= limit:
            print(f"   ⚠️  Límite alcanzado: {total_samples} muestras")
            break
        
        # Mostrar progreso cada 100 batches
        if len(X_list) % 100 == 0:
            print(f"   Procesados {len(X_list)} batches ({total_samples} muestras)...")
    
    # Concatenar todos los batches
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # Aplicar límite exacto si se especificó
    if limit is not None and X.shape[0] > limit:
        X = X[:limit]
        y = y[:limit]
    
    print(f"✅ {dataset_name}: {X.shape[0]:,} muestras")
    print(f"   Shape: {X.shape}")
    
    return X, y


def optimize_dataset_performance(dataset, prefetch_buffer=tf.data.AUTOTUNE):
    """
    Optimiza el rendimiento del dataset con prefetching y caching.
    
    Parámetros
    ----------
    dataset : tf.data.Dataset
        Dataset a optimizar
    prefetch_buffer : int, opcional
        Tamaño del buffer de prefetch
        Por defecto: tf.data.AUTOTUNE
    
    Retorna
    -------
    tf.data.Dataset
        Dataset optimizado
    """
    return dataset.cache().prefetch(buffer_size=prefetch_buffer)


if __name__ == "__main__":
    # Ejemplo de uso
    print("="*60)
    print("EJEMPLO DE USO DEL DATALOADER")
    print("="*60 + "\n")
    
    # Buscar automáticamente la ruta de datos
    import os
    
    # Buscar la estructura correcta del dataset
    search_path = "."
    data_path = None
    
    for root, dirs, files in os.walk(search_path):
        if 'train' in dirs and 'test' in dirs:
            data_path = root
            break
    
    if data_path:
        print(f"✅ Datos encontrados en: {data_path}\n")
        
        # Crear dataloaders con configuración personalizada
        train_ds, val_ds, test_ds, classes = create_oct_dataloaders(
            data_path=data_path,
            img_size=(224, 224),
            batch_size=32,
            train_subset_fraction=0.5,  # Usar solo 50% para este ejemplo
            verbose=True
        )
        
        # Ejemplo: iterar sobre un batch
        print("\n📐 Verificando forma del primer batch:")
        for images, labels in train_ds.take(1):
            print(f"   • Images shape: {images.shape}")
            print(f"   • Labels shape: {labels.shape}")
            print(f"   • Rango de valores: [{tf.reduce_min(images).numpy():.0f}, {tf.reduce_max(images).numpy():.0f}]")
        
        print("\n✅ DataLoaders listos para usar!")
        
    else:
        print("❌ No se encontró estructura de datos con carpetas 'train' y 'test'")
        print("\nPara usar este módulo, importa la función:")
        print("   from oct_dataloader import create_oct_dataloaders")
