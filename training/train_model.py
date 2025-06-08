import os
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# virifica se o treinamento tá usando a GPU
def verify_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPUs disponíveis: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.warning("Nenhuma GPU encontrada. O treinamento será mais lento.")

# cria o modelo
def create_model():
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

# para salvar o modelo e colocar a extensão certa
def save_model_safe(model, output_path_no_ext: str):
    
    # Adiciona extensão .h5
    save_path = output_path_no_ext + '.h5'
    # Garante que a pasta existe
    carpeta = os.path.dirname(save_path)
    if carpeta and not os.path.exists(carpeta):
        os.makedirs(carpeta, exist_ok=True)
        logger.info(f"Diretório criado: {carpeta}")
        
    # Salva o modelo
    model.save(save_path)
    logger.info(f"Modelo salvo em: {save_path}")
    
def main():

    # configurAÇÃO DO modelo
    IMG_SIZE = (224, 224)        # tamanho da imagem de entreda
    BATCH_SIZE = 32
    EPOCHS = 20                  # Pode ajustar conforme necessidade
    DATASET_PATH = "dataset"     # Pasta contendo subpastas de classes
    MODEL_OUTPUT_PATH = "output/model" # onde o modelo será salvo

    THRESHOLD = 0.5

    # VERIFICAR GPU
    verify_gpu()
    
    # CRIAR GERADORES DE DADOS (treinamento e validação)
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
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
        logger.error(f"Erro ao carregar dados: {e}")
        return

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
    
    logger.info("Salvando modelo...")
    save_model_safe(model, MODEL_OUTPUT_PATH)
    logger.info("Treinamento concluído com sucesso!")

if __name__ == "__main__":
    main()
