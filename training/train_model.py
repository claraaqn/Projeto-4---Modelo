import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_gpu():
    """Verifica se o TensorFlow está usando GPU"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPUs disponíveis: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.warning("Nenhuma GPU encontrada. O treinamento será mais lento.")

def create_model():
    """Cria o modelo de classificação"""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Configurações
    IMG_SIZE = (224, 224)  # Corrigido para corresponder ao MobileNetV2
    BATCH_SIZE = 32
    EPOCHS = 20  # Reduzido para teste, aumente depois
    DATASET_PATH = "dataset"
    MODEL_OUTPUT_PATH = os.path.join('..', 'output', 'model')
    
    # Verificar GPU
    verify_gpu()
    
    # Criar diretórios de saída
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    
    # Criar geradores de dados
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    logger.info("Criando geradores de dados...")
    try:
        train_generator = train_datagen.flow_from_directory(
            DATASET_PATH,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='training'
        )

        val_generator = train_datagen.flow_from_directory(
            DATASET_PATH,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='validation'
        )
    except Exception as e:
        logger.error(f"Erro ao cargar dados: {e}")
        return

    # Criar e treinar modelo
    logger.info("Criando modelo...")
    model = create_model()
    
    logger.info("Iniciando treinamento...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    # Salvar modelo
    logger.info("Salvando modelo...")
    model.save(f'{MODEL_OUTPUT_PATH}.h5')
    logger.info("Treinamento concluído com sucesso!")

if __name__ == "__main__":
    main()