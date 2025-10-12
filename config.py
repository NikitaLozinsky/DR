import json
import pandas as pd
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
    'TOP_FEATURES': 17
}


def calculate_real_thresholds(csv_path):
    """Автоматически вычисляет реальные пороги на основе данных"""
    try:
        df = pd.read_csv(csv_path)

        # Разделяем на здоровых и больных
        healthy = df[df['label'] == 0]
        sick = df[df['label'] == 1]

        # Вычисляем пороги как среднее между медианами классов
        thresholds = {
            'microaneurysms_count': (healthy['microaneurysms_count'].median() +
                                     sick['microaneurysms_count'].median()) / 2,
            'exudates_area': (healthy['exudates_area'].median() +
                              sick['exudates_area'].median()) / 2,
            'vessel_length': (healthy['vessel_length'].median() +
                              sick['vessel_length'].median()) / 2,
            'dark_to_light_ratio': (healthy['dark_to_light_ratio'].median() +
                                    sick['dark_to_light_ratio'].median()) / 2,
            'entropy_mean': (healthy['entropy_mean'].median() +
                             sick['entropy_mean'].median()) / 2
        }

        return thresholds, healthy, sick

    except Exception as e:
        print(f"Ошибка вычисления порогов: {e}. Использую значения по умолчанию.")
        # Возвращаем разумные значения по умолчанию на основе твоих данных
        return {
            'microaneurysms_count': 18000,
            'exudates_area': 11800,
            'vessel_length': 1900,
            'dark_to_light_ratio': 1.5,
            'entropy_mean': 1.7
        }, None, None


def calculate_healthy_ranges(healthy_df):
    """Вычисляет здоровые диапазоны на основе 5-95 перцентилей"""
    if healthy_df is None or len(healthy_df) == 0:
        return {
            'microaneurysms_count': (16000, 19000),
            'exudates_area': (11000, 13000),
            'vessel_length': (1500, 2500),
            'dark_to_light_ratio': (1.3, 1.7),
            'entropy_mean': (1.4, 2.0)
        }

    return {
        'microaneurysms_count': (
            healthy_df['microaneurysms_count'].quantile(0.05),
            healthy_df['microaneurysms_count'].quantile(0.95)
        ),
        'exudates_area': (
            healthy_df['exudates_area'].quantile(0.05),
            healthy_df['exudates_area'].quantile(0.95)
        ),
        'vessel_length': (
            healthy_df['vessel_length'].quantile(0.05),
            healthy_df['vessel_length'].quantile(0.95)
        ),
        'dark_to_light_ratio': (
            healthy_df['dark_to_light_ratio'].quantile(0.05),
            healthy_df['dark_to_light_ratio'].quantile(0.95)
        ),
        'entropy_mean': (
            healthy_df['entropy_mean'].quantile(0.05),
            healthy_df['entropy_mean'].quantile(0.95)
        )
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

            # Объединяем сохраненную конфигурацию со значениями по умолчанию
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

# Вычисляем РЕАЛЬНЫЕ пороги на основе данных
try:
    features_csv = Path(current_config['DATASET_DIR']) / 'enhanced_retinopathy_features.csv'
    FEATURE_THRESHOLDS, healthy_df, sick_df = calculate_real_thresholds(features_csv)
    HEALTHY_FEATURE_RANGES = calculate_healthy_ranges(healthy_df)
except Exception as e:
    print(f"Не удалось вычислить пороги: {e}. Использую запасные значения.")
    # Значения на основе данных с прошлых моделей
    FEATURE_THRESHOLDS = {
        'microaneurysms_count': 18000,
        'exudates_area': 11800,
        'vessel_length': 1900,
        'dark_to_light_ratio': 1.5,
        'entropy_mean': 1.7
    }
    HEALTHY_FEATURE_RANGES = {
        'microaneurysms_count': (16000, 19000),
        'exudates_area': (11000, 13000),
        'vessel_length': (1500, 2500),
        'dark_to_light_ratio': (1.3, 1.7),
        'entropy_mean': (1.4, 2.0)
    }

# Экспортируем параметры
IMAGE_SIZE = tuple(current_config['IMAGE_SIZE'])
THRESHOLDS = current_config['THRESHOLDS']
DATASET_DIR = Path(current_config['DATASET_DIR'])
RESULTS_DIR = Path(current_config['RESULTS_DIR'])
IMAGES_DIR = Path(current_config['IMAGES_DIR'])
RANDOM_STATE = current_config['RANDOM_STATE']
TEST_SIZE = current_config['TEST_SIZE']
TOP_FEATURES = current_config['TOP_FEATURES']

# Выводим информацию о порогах для отладки
# print("=" * 50)
# print("РАСЧЁТНЫЕ ПОРОГИ ДИАГНОСТИКИ:")
# for feature, threshold in FEATURE_THRESHOLDS.items():
#     print(f"  {feature}: {threshold:.2f}")
# print("=" * 50)