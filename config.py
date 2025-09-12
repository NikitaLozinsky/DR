import os
from pathlib import Path

# Базовые пути
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / 'dataset'
RESULTS_DIR = BASE_DIR / 'results'
IMAGES_DIR = DATASET_DIR / 'images'

# Создание директорий если они не существуют
for directory in [DATASET_DIR, RESULTS_DIR, IMAGES_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Параметры обработки изображений
IMAGE_SIZE = (128, 128)
THRESHOLDS = {
    'dark': 30,
    'light': 200
}

# Параметры модели
RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_FEATURES = 15