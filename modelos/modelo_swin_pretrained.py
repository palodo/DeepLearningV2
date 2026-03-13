import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tfswin

def create_pretrained_swin(input_shape=(128, 128, 3), num_classes=4, dropout_rate=0.3):
    """
    Crea un modelo Swin Transformer utilizando pesos pre-entrenados (Transfer Learning).
    Nota: Swin suele requerir imágenes de 3 canales (RGB). Si tus datos son escala de grises,
    se deben convertir a 3 canales antes de pasarlos al modelo.
    """
    
    # Base del modelo Swin (Tiny o Base) pre-entrenado en ImageNet
    # Tiny es más ligero y adecuado para experimentos rápidos
    base_model = tfswin.SwinTransformerTiny224(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Congelar el modelo base (opcional, se puede descongelar para fine-tuning)
    base_model.trainable = False
    
    # Añadir clasificador personalizado
    inputs = layers.Input(shape=input_shape)
    
    # Preprocesamiento: Swin espera valores en cierto rango (normalmente [0, 255] o [0, 1] dependiendo de la implementación)
    # tfswin maneja la normalización internamente si se usa su función de preprocesamiento, 
    # pero aquí asumimos que el dataloader entrega [0, 255] y re-escalamos si es necesario.
    x = base_model(inputs, training=False)
    
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
