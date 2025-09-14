import argparse
import os
import sys
import logging
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from config import RESULTS_DIR, DATASET_DIR
from preprocess_images import ImageProcessor

# Настройка логирования только для ошибок
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RetinopathyDiagnosis:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = None
        self.processor = ImageProcessor()
        self.is_loaded = False

    def load_resources(self):
        """Загрузка обученных моделей и ресурсов"""
        try:
            # Проверяем существование файлов
            model_path = RESULTS_DIR / 'best_model_enhanced.pkl'
            scaler_path = RESULTS_DIR / 'scaler_enhanced.pkl'
            features_path = DATASET_DIR / 'top_features.csv'

            if not model_path.exists():
                logger.error(f"Файл модели не найден: {model_path}")
                return False
            if not scaler_path.exists():
                logger.error(f"Файл scaler не найден: {scaler_path}")
                return False
            if not features_path.exists():
                logger.error(f"Файл с признаками не найден: {features_path}")
                return False

            # Загружаем ресурсы
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.features = pd.read_csv(features_path).iloc[:, 0].tolist()

            # Проверяем, был ли scaler обучен
            if not hasattr(self.scaler, 'mean_') or not hasattr(self.scaler, 'scale_'):
                logger.error("Scaler не был правильно обучен")
                return False

            self.is_loaded = True
            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки ресурсов: {e}")
            return False

    def analyze_image(self, image_path):
        """Анализ одного изображения"""
        if not self.is_loaded:
            logger.error("Ресурсы не загружены")
            return None

        try:
            features = self.processor.process_image(image_path)
            if not features:
                logger.error(f"Не удалось извлечь признаки из изображения: {image_path}")
                return None

            # Создаем DataFrame с правильным порядком признаков
            feature_values = []
            for feature_name in self.features:
                if feature_name in features:
                    feature_values.append(features[feature_name])
                else:
                    logger.warning(f"Признак {feature_name} отсутствует в извлеченных признаках. Использую 0.")
                    feature_values.append(0)

            # Создаем DataFrame с явными именами столбцов
            features_df = pd.DataFrame([feature_values], columns=self.features)

            # Масштабирование и предсказание
            features_scaled = self.scaler.transform(features_df)

            # Подавляем предупреждения о feature names
            import warnings
            from sklearn.exceptions import DataConversionWarning
            warnings.filterwarnings(action='ignore', category=UserWarning)

            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]

            return {
                'prediction': prediction,
                'confidence': probability[1] if prediction == 1 else probability[0],
                'features': features
            }

        except Exception as e:
            logger.error(f"Ошибка анализа изображения: {e}")
            return None


def clean_path(path_string):
    """Очистка пути от кавычек и лишних пробелов"""
    path_string = path_string.strip().strip('"').strip("'")
    # Удаляем возможные лишние кавычки
    if path_string.startswith('"') and path_string.endswith('"'):
        path_string = path_string[1:-1]
    return path_string


def interactive_mode(diagnosis):
    """Интерактивный режим работы"""
    print("=" * 60)
    print("ДИАГНОСТИКА ДИАБЕТИЧЕСКОЙ РЕТИНОПАТИИ")
    print("=" * 60)

    if not diagnosis.is_loaded:
        print("Ошибка: Модель не загружена. Убедитесь, что вы обучили модель перед использованием.")
        print("Запустите: python run_pipeline.py")
        return

    print("Введите путь к изображению для анализа или 'exit' для выхода")

    while True:
        try:
            user_input = input("\nПуть к изображению: ").strip()

            if user_input.lower() in ['exit', 'quit', 'выход']:
                print("Завершение работы...")
                break

            # Проверяем, не пустой ли ввод
            if not user_input:
                continue

            image_path = clean_path(user_input)

            if not os.path.exists(image_path):
                print(f"Ошибка: файл '{image_path}' не существует!")
                continue

            result = diagnosis.analyze_image(image_path)
            if result:
                print(f"\nРезультат: {'Есть ретинопатия' if result['prediction'] == 1 else 'Нет ретинопатии'}")
                print(f"Уверенность: {result['confidence']:.2%}")

                # Дополнительная информация
                if result['prediction'] == 1:
                    print("\nКлючевые признаки ретинопатии:")
                    if result['features'].get('dark_objects_count', 0) > 2:
                        print(
                            f"- Обнаружены потенциальные микроаневризмы/кровоизлияния: {result['features']['dark_objects_count']}")
                    if result['features'].get('bright_objects_count', 0) > 1:
                        print(f"- Обнаружены потенциальные экссудаты: {result['features']['bright_objects_count']}")
                    if result['features'].get('glcm_contrast', 0) > 100:
                        print("- Выявлена неоднородность текстуры сетчатки")
                else:
                    print("\nПризнаки здоровой сетчатки:")
                    if result['features'].get('dark_objects_count', 0) <= 2:
                        print("- Микроаневризмы/кровоизлияния не обнаружены")
                    if result['features'].get('bright_objects_count', 0) <= 1:
                        print("- Экссудаты не обнаружены")
                    if result['features'].get('glcm_homogeneity', 0) > 0.4:
                        print("- Текстура сетчатки однородна")
            else:
                print("Не удалось проанализировать изображение")

        except KeyboardInterrupt:
            print("\nЗавершение работы...")
            break
        except Exception as e:
            print(f"Произошла непредвиденная ошибка: {e}")


def main():
    """Основная функция диагностики"""
    parser = argparse.ArgumentParser(description='Диагностика диабетической ретинопатии')
    parser.add_argument('--image', '-i', help='Путь к изображению для анализа')
    args = parser.parse_args()

    diagnosis = RetinopathyDiagnosis()
    if not diagnosis.load_resources():
        print("Ошибка загрузки модели. Убедитесь, что вы обучили модель перед использованием.")
        print("Запустите: python run_pipeline.py")
        sys.exit(1)

    # Если указан путь к изображению как аргумент
    if args.image:
        image_path = clean_path(args.image)
        if not os.path.exists(image_path):
            print(f"Ошибка: файл '{image_path}' не существует!")
            sys.exit(1)

        result = diagnosis.analyze_image(image_path)
        if result:
            print(f"Результат: {'Есть ретинопатия' if result['prediction'] == 1 else 'Нет ретинопатии'}")
            print(f"Уверенность: {result['confidence']:.2%}")
        else:
            print("Не удалось проанализировать изображение")
            sys.exit(1)
    else:
        # Интерактивный режим
        interactive_mode(diagnosis)


if __name__ == "__main__":
    main()