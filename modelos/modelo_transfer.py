"""
Transfer Learning utilizando VGG16 para Clasificación de Imágenes OCT
==================================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16

def create_transfer_model(input_shape=(128, 128, 3), num_classes=4):
    """
    Crea un modelo basado en VGG16 preentrenado.
    """
    # Base preentrenada (ImageNet)
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Congelar la base
    base_model.trainable = False
    
    # Añadir nuevas capas superiores
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='TransferVGG16')
    
    return model

def compile_model(model, learning_rate=0.001):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def print_model_summary(model):
    model.summary()
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"\n📊 Parámetros totales: {total_params:,}")
    print(f"📈 Parámetros entrenables: {trainable_params:,}")

if __name__ == "__main__":
    model = create_transfer_model()
    compile_model(model)
    print_model_summary(model)
