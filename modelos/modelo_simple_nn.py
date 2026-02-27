"""
Red Neuronal Simple para Clasificación de Imágenes OCT
========================================================

Red neuronal con el mínimo de parámetros y capas posible.

Arquitectura:
- Input: Imágenes 64x64 en escala de grises (4,096 píxeles)
- Flatten: Convierte imagen 2D a vector 1D
- Dense(32): Capa oculta con 32 neuronas y activación ReLU
- Dense(4): Capa de salida con 4 clases (softmax)

Total de parámetros: ~131,236
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_simple_nn(input_shape=(64, 64, 1), num_classes=4, hidden_units=32):
    """
    Crea una red neuronal simple con el mínimo de capas.
    
    Args:
        input_shape: Forma de las imágenes de entrada (height, width, channels)
        num_classes: Número de clases de salida
        hidden_units: Número de neuronas en la capa oculta
    
    Returns:
        modelo: Red neuronal compilada
    """
    
    model = keras.Sequential([
        # Capa de entrada
        layers.Input(shape=input_shape),
        
        # Aplanar imagen 2D a vector 1D
        layers.Flatten(),
        
        # UNA SOLA capa oculta (lo mínimo posible)
        layers.Dense(hidden_units, activation='relu', name='hidden'),
        
        # Capa de salida
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='SimpleNN')
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compila el modelo con optimizador y función de pérdida.
    
    Args:
        model: Modelo a compilar
        learning_rate: Tasa de aprendizaje para el optimizador
    
    Returns:
        model: Modelo compilado
    """
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def print_model_summary(model):
    """
    Imprime un resumen detallado del modelo.
    
    Args:
        model: Modelo de Keras
    """
    
    print("\n" + "=" * 70)
    print(" " * 20 + "📊 ARQUITECTURA DEL MODELO")
    print("=" * 70)
    
    model.summary()
    
    # Contar parámetros
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    print("\n" + "=" * 70)
    print("📈 RESUMEN DE PARÁMETROS")
    print("=" * 70)
    print(f"   • Parámetros totales: {total_params:,}")
    print(f"   • Parámetros entrenables: {trainable_params:,}")
    print(f"   • Capas: {len(model.layers)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Ejemplo de uso
    print("Creando red neuronal simple...")
    
    # Crear modelo
    model = create_simple_nn(input_shape=(64, 64, 1), num_classes=4, hidden_units=32)
    
    # Compilar modelo
    model = compile_model(model, learning_rate=0.001)
    
    # Mostrar resumen
    print_model_summary(model)
    
    print("\n✅ Modelo creado correctamente")
    print(f"   Nombre: {model.name}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
