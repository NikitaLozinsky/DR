import json
from pathlib import Path

# Значения по умолчанию
DEFAULT_CONFIG = {
    'IMAGE_SIZE': [128, 128],
    'THRESHOLDS': {'dark': 30, 'light': 200},
    'DATASET_DIR': 'dataset',
    'RESULTS_DIR': 'results',
    'IMAGES_DIR': 'dataset/images',
    'RANDOM_STATE': 42,
    'TEST_SIZE': 0.2,
    'TOP_FEATURES': 100
}

# Нормативные значения для диагностики
FEATURE_THRESHOLDS = {
    'microaneurysms_count': 5,      # Если больше этого значения - признак ретинопатии
    'exudates_area': 30,            # Если больше этого значения - признак ретинопатии
    'vessel_length': 200,           # Если меньше этого значения - возможный признак
    'dark_to_light_ratio': 0.1,     # Если больше этого значения - признак ретинопатии
    'entropy_mean': 4.0             # Если меньше этого значения - возможный признак
}

HEALTHY_FEATURE_RANGES = {
    'microaneurysms_count': (0, 5),
    'exudates_area': (0, 30),
    'vessel_length': (200, 1000),
    'dark_to_light_ratio': (0, 0.1),
    'entropy_mean': (4.0, 8.0)
}
# Файл для сохранения конфигурации
CONFIG_FILE = Path(__file__).parent / 'config.json'


def load_config():
    """Загружает конфигурацию из файла или использует значения по умолчанию"""
    base_dir = Path(__file__).parent

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)

            # Объединяем сохраненную конфигурацию с значениями по умолчанию
            config = DEFAULT_CONFIG.copy()
            config.update(saved_config)

            # Преобразуем относительные пути в абсолютные
            for key in ['DATASET_DIR', 'RESULTS_DIR', 'IMAGES_DIR']:
                path_value = config[key]
                if not Path(path_value).is_absolute():
                    config[key] = str(base_dir / path_value)
                else:
                    config[key] = path_value

            return config
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}. Использую значения по умолчанию.")

    # Возвращаем копию значений по умолчанию с абсолютными путями
    config = DEFAULT_CONFIG.copy()
    for key in ['DATASET_DIR', 'RESULTS_DIR', 'IMAGES_DIR']:
        config[key] = str(base_dir / config[key])

    return config


# Загружаем текущую конфигурацию
current_config = load_config()

# Экспортируем параметры
IMAGE_SIZE = tuple(current_config['IMAGE_SIZE'])
THRESHOLDS = current_config['THRESHOLDS']
DATASET_DIR = Path(current_config['DATASET_DIR'])
RESULTS_DIR = Path(current_config['RESULTS_DIR'])
IMAGES_DIR = Path(current_config['IMAGES_DIR'])
RANDOM_STATE = current_config['RANDOM_STATE']
TEST_SIZE = current_config['TEST_SIZE']
TOP_FEATURES = current_config['TOP_FEATURES']