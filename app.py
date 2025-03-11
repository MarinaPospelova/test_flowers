from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.neighbors import NearestNeighbors
import os
from PIL import Image
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Загрузка модели VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
model.compile(run_eagerly=True)

def load_image_without_metadata(img_path):
    """Загрузка и предобработка изображения."""
    img = Image.open(img_path)
    img = img.convert("RGB")  # Убедитесь, что изображение в формате RGB
    img = img.resize((224, 224))  # Измените размер
    img = image.img_to_array(img)
    img = preprocess_input(img)
    return img

def load_test_images(data_dir):
    """Загрузка тестовых изображений из директории."""
    images = []
    image_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                img_path = os.path.join(root, file)
                try:
                    img = load_image_without_metadata(img_path)
                    images.append(img)
                    image_paths.append(img_path)
                except Exception as e:
                    logging.error(f"Failed to load {img_path}: {e}")
    return np.array(images), image_paths

def extract_features(img_array, model):
    """Извлечение признаков из изображений с использованием модели."""
    features = model.predict(img_array)
    features = features.reshape(features.shape[0], -1)
    return features

def find_similar_images(query_img, test_features, test_image_paths, n_neighbors=5):
    """Поиск похожих изображений с использованием NearestNeighbors."""
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    neighbors.fit(test_features)
    distances, indices = neighbors.kneighbors(query_img)
    return [(test_image_paths[idx], 1 - distances[0][i]) for i, idx in enumerate(indices[0])]

# Загрузка тестовой выборки
test_data_dir = 'flowers'
logging.debug("Loading test images...")
test_images, test_image_paths = load_test_images(test_data_dir)
if len(test_images) == 0:
    raise ValueError("No images found in the test directory.")
logging.debug(f"Loaded {len(test_images)} images.")

logging.debug("Extracting features...")
test_features = extract_features(test_images, model)
logging.debug("Features extracted.")

@app.route('/')
def index():
    """Главная страница с формой загрузки изображения."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Обработка запроса на поиск похожих изображений."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty file"}), 400

    try:
        # Загрузка и обработка загруженного изображения
        img = Image.open(file)  # Используем Image.open для открытия файла
        img = img.resize((224, 224))  # Изменяем размер
        img_array = load_image_without_metadata(img)

        # Извлечение признаков
        query_features = extract_features(np.array([img_array]), model)

        # Поиск похожих изображений
        similar_images = find_similar_images(query_features, test_features, test_image_paths)

        return jsonify(similar_images)
    except Exception as e:
        logging.error(f"Error processing the image: {e}", exc_info=True)
        return jsonify({"error": "Failed to process the image"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
