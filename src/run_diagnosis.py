import argparse
import os
import sys
import logging
import pandas as pd
import joblib
from pathlib import Path
from config import RESULTS_DIR, DATASET_DIR
from preprocess_images import ImageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetinopathyDiagnosis:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = None
        self.processor = ImageProcessor()

    def load_resources(self):
        """Загрузка обученных моделей и ресурсов"""
        try:
            self.model = joblib.load(RESULTS_DIR / 'best_model_enhanced.pkl')
            self.scaler = joblib.load(RESULTS_DIR / 'scaler_enhanced.pkl')
            self.features = pd.read_csv(DATASET_DIR / 'top_features.csv').iloc[:, 0].tolist()
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки ресурсов: {e}")
            return False

    def analyze_image(self, image_path):
        """Анализ одного изображения"""
        try:
            features = self.processor.process_image(image_path)
            if not features:
                return None

            # Отбор нужных признаков
            selected_features = {f: features[f] for f in self.features if f in features}
            features_df = pd.DataFrame([selected_features])

            # Масштабирование и предсказание
            features_scaled = self.scaler.transform(features_df)
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
    return path_string.strip().strip('"').strip("'")


def interactive_mode(diagnosis):
    """Интерактивный режим работы"""
    print("=" * 60)
    print("ДИАГНОСТИКА ДИАБЕТИЧЕСКОЙ РЕТИНОПАТИИ")
    print("=" * 60)
    print("Введите путь к изображению для анализа или 'exit' для выхода")

    while True:
        user_input = input("\nПуть к изображению: ").strip()

        if user_input.lower() in ['exit', 'quit', 'выход']:
            print("Завершение работы...")
            break

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
                print("\nПризнаки ретинопатии:")
                if result['features'].get('microaneurysms_count', 0) > 0:
                    print(f"- Обнаружены микроаневризмы: {result['features']['microaneurysms_count']}")
                if result['features'].get('exudates_area', 0) > 0:
                    print(f"- Обнаружены экссудаты: {result['features']['exudates_area']}")
            else:
                print("\nПризнаки здоровой сетчатки:")
                if result['features'].get('microaneurysms_count', 0) == 0:
                    print("- Микроаневризмы не обнаружены")
                if result['features'].get('exudates_area', 0) == 0:
                    print("- Экссудаты не обнаружены")
        else:
            print("Не удалось проанализировать изображение")


def main():
    """Основная функция диагностики"""
    parser = argparse.ArgumentParser(description='Диагностика диабетической ретинопатии')
    parser.add_argument('--image', '-i', help='Путь к изображению для анализа')
    args = parser.parse_args()

    diagnosis = RetinopathyDiagnosis()
    if not diagnosis.load_resources():
        print("Ошибка загрузки модели. Убедитесь, что вы обучили модель перед использованием.")
        sys.exit(1)

    # Если указан путь к изображению как аргумент
    if args.image:
        result = diagnosis.analyze_image(args.image)
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