import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parâmetro de threshold (taxa de decisão)
THRESHOLD = 0.5

# Caminho do modelo salvo
MODEL_PATH = "output\model.h5"

def load_model():
    """Carrega o modelo salvo."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")
    logger.info(f"Carregando modelo de {MODEL_PATH}...")
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(img_path, target_size=(224, 224)):
    """Pré-processa uma imagem para predição com MobileNetV2."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normaliza como no treinamento
    return np.expand_dims(img_array, axis=0)

def predict_image(model, img_path):
    """Realiza a predição da imagem usando o modelo carregado."""
    input_img = preprocess_image(img_path)
    prob = model.predict(input_img)[0][0]
    predicted_class = int(prob > THRESHOLD)
    logger.info(f"Probabilidade: {prob:.4f} | Classe prevista: {predicted_class} (threshold={THRESHOLD})")
    return predicted_class, prob

if __name__ == "__main__":
    model = load_model()
    
    # Caminho da imagem de teste (ajuste conforme necessário)
    img_teste = 'testes\\foto para teste.jpeg'
    
    # Faz predição
    try:
        classe, probabilidade = predict_image(model, img_teste)
        print(f"Classe: {classe} | Probabilidade: {probabilidade:.4f}")
    except Exception as e:
        logger.error(f"Erro ao prever imagem: {e}")
