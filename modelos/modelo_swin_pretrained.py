import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tfswin

def create_pretrained_swin(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.3, use_augmentation=False):
    """
    Crea un modelo Swin Transformer utilizando pesos pre-entrenados.
    """
    
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Data Augmentation integrado en el modelo (se activa solo durante model.fit)
    if use_augmentation:
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.15)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomContrast(0.1)(x)
        x = layers.RandomBrightness(0.1)(x)

    # Base del modelo Swin
    base_model = tfswin.SwinTransformerTiny224(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Desbloqueado para entrenamiento completo
    base_model.trainable = True
    
    x = base_model(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="PretrainedSwinOCT")
    
    return model

def compile_swin_model(model, learning_rate=0.0001, metrics=['accuracy']):
    """
    Compila el modelo. Usamos un LR más bajo para Transfer Learning.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=metrics
    )
    return model

def get_callbacks(patience_stop=8, patience_lr=4, factor_lr=0.5):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
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
        )
    ]
    return callbacks
