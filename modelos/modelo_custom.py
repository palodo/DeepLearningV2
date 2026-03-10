"""
Modelo CNN Custom con Soporte para Learning Rate y Schedulers
============================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_custom_cnn(input_shape=(128, 128, 1), num_classes=4, dropout_rate=0.4):
    """
    Crea una CNN con BatchNormalization y Dropout configurable.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Bloque 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Bloque 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Bloque 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Clasificador
        layers.Flatten(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ], name='CustomProCNN')
    
    return model

def compile_custom_model(model, learning_rate=0.001, metrics=['accuracy']):
    """
    Compila el modelo permitiendo pasar el learning rate y métricas.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )
    return model

def get_callbacks(patience_stop=5, patience_lr=3, factor_lr=0.2):
    """
    Retorna los callbacks para EarlyStopping y ReduceLROnPlateau.
    """
    callbacks = [
        # Detiene el entrenamiento si no mejora
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience_stop,
            restore_best_weights=True,
            verbose=1
        ),
        # Baja el Learning Rate automáticamente si el aprendizaje se estanca
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=factor_lr,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1
        )
    ]
    return callbacks
