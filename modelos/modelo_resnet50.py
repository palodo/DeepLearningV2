"""
Transfer Learning con ResNet50 para Clasificación de Imágenes OCT
================================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50

def create_resnet_model(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.4):
    """
    Crea un modelo basado en ResNet50 preentrenado.
    Nota: ResNet50 requiere que las imágenes tengan 3 canales (RGB).
    """
    # Base preentrenada (ImageNet)
    # Importante: weights='imagenet' descarga los pesos si no están en cache
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Congelar la base por defecto (Transfer Learning inicial)
    base_model.trainable = False
    
    # Añadir nuevas capas superiores
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ], name='ResNet50_Transfer')
    
    return model

def compile_resnet_model(model, learning_rate=0.001, metrics=['accuracy']):
    """
    Compila el modelo permitiendo pasar el learning rate y métricas.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )
    return model

def get_resnet_callbacks(patience_stop=6, patience_lr=3, factor_lr=0.2, monitor_stop='val_loss', weights_path='modelos/best_resnet50.h5'):
    """
    Retorna los callbacks para EarlyStopping, ReduceLROnPlateau y ModelCheckpoint.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_stop,
            patience=patience_stop,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=factor_lr,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=weights_path,
            monitor=monitor_stop,
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]
    return callbacks
