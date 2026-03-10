"""
Red Neuronal Convolucional (CNN) para Clasificación de Imágenes OCT
================================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_cnn(input_shape=(128, 128, 1), num_classes=4):
    """
    Crea una arquitectura CNN básica.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Bloque Conv 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Bloque Conv 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Bloque Conv 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Aplanado y Capas Densas
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Para evitar sobreajuste
        layers.Dense(num_classes, activation='softmax')
    ], name='BasicCNN')
    
    return model

def compile_model(model, learning_rate=0.001, metrics=['accuracy']):
    """
    Compila el modelo con el optimizador Adam y parámetros configurables.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )
    return model

def print_model_summary(model):
    model.summary()
    total_params = model.count_params()
    print(f"\n✅ Parámetros totales: {total_params:,}")

if __name__ == "__main__":
    model = create_cnn()
    compile_model(model)
    print_model_summary(model)
