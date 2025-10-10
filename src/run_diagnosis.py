import argparse
import os
import sys
import logging
import pandas as pd
import joblib
from pathlib import Path
from config import RESULTS_DIR, DATASET_DIR, FEATURE_THRESHOLDS, HEALTHY_FEATURE_RANGES
from preprocess_images import ImageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetinopathyDiagnosis:
    def __init__(self):
        self.origin_model = None
        self.goty_model = None
        self.goty_scaler = None
        self.goty_features = None
        self.current_mode = 'origin'  # Режим по умолчанию
        self.processor = ImageProcessor()

    def load_resources(self):
        """Загрузка обученных моделей и ресурсов"""
        try:
            # Загружаем обе модели
            self.origin_model = joblib.load(RESULTS_DIR / 'origin_model.pkl')
            self.goty_model = joblib.load(RESULTS_DIR / 'best_model_enhanced.pkl')
            self.goty_scaler = joblib.load(RESULTS_DIR / 'scaler_enhanced.pkl')
            self.goty_features = pd.read_csv(DATASET_DIR / 'top_features.csv').iloc[:, 0].tolist()
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

            if self.current_mode == 'origin':
                # Используем только 2 признака из ТЗ
                selected_features = {
                    'mean_brightness': features['mean_brightness'],
                    'std_brightness': features['std_brightness']
                }
                features_df = pd.DataFrame([selected_features])

                # Для origin модели не нужно масштабирование
                prediction = self.origin_model.predict(features_df)[0]
                probability = self.origin_model.predict_proba(features_df)[0]
            else:
                # Режим goty - все признаки
                selected_features = {f: features[f] for f in self.goty_features if f in features}
                features_df = pd.DataFrame([selected_features])
                features_scaled = self.goty_scaler.transform(features_df)
                prediction = self.goty_model.predict(features_scaled)[0]
                probability = self.goty_model.predict_proba(features_scaled)[0]

            return {
                'prediction': prediction,
                'confidence': probability[1] if prediction == 1 else probability[0],
                'features': features,
                'mode': self.current_mode
            }

        except Exception as e:
            logger.error(f"Ошибка анализа изображения в режиме {self.current_mode}: {e}")
            return None

    def switch_mode(self):
        """Переключение режима диагностики"""
        self.current_mode = 'goty' if self.current_mode == 'origin' else 'origin'
        return self.current_mode


def clean_path(path_string):
    """Очистка пути от кавычек и лишних пробелов"""
    return path_string.strip().strip('"').strip("'")


def interactive_mode(diagnosis):
    """Интерактивный режим работы"""
    print("=" * 60)
    print("ДИАГНОСТИКА ДИАБЕТИЧЕСКОЙ РЕТИНОПАТИИ")
    print("=" * 60)
    print("Команды:")
    print("- Введите путь к изображению для анализа")
    print("- 'mode' - переключить режим диагностики")
    print("- 'exit' - выйти из программы")
    print(f"\nТекущий режим: {diagnosis.current_mode}")
    print("  origin: только mean_brightness и std_brightness (по ТЗ)")
    print("  goty: все расширенные признаки")

    while True:
        user_input = input(f"\n[{diagnosis.current_mode}] Путь к изображению: ").strip()

        if user_input.lower() in ['exit', 'quit', 'выход']:
            print("Завершение работы...")
            break

        # Обработка команды смены режима
        if user_input.lower() == 'mode':
            new_mode = diagnosis.switch_mode()
            mode_description = "только mean_brightness и std_brightness (ТЗ)" if new_mode == 'origin' else "все расширенные признаки"
            print(f"✅ Режим изменен на: {new_mode} ({mode_description})")
            continue

        image_path = clean_path(user_input)

        if not os.path.exists(image_path):
            print(f"❌ Ошибка: файл '{image_path}' не существует!")
            continue

        result = diagnosis.analyze_image(image_path)
        if result:
            status = "Есть ретинопатия" if result['prediction'] == 1 else "Нет ретинопатии"
            print(f"\n📊 Режим: {result['mode']}")
            print(f"🎯 Результат: {status}")
            print(f"📈 Уверенность: {result['confidence']:.2%}")

            # Показываем используемые признаки в зависимости от режима
            if result['mode'] == 'origin':
                print(f"🔧 Используемые признаки: mean_brightness, std_brightness")
                print(f"   - mean_brightness: {result['features']['mean_brightness']:.2f}")
                print(f"   - std_brightness: {result['features']['std_brightness']:.2f}")
            else:
                print(f"🔧 Используемые признаки: все {len(result['features'])} признаков")
                # Можно показать топ-5 самых важных признаков
                top_features = ['microaneurysms_count', 'exudates_area', 'vessel_length',
                                'dark_to_light_ratio', 'entropy_mean']
                print("   Самые важные признаки:")
                for feature in top_features:
                    if feature in result['features']:
                        print(f"   - {feature}: {result['features'][feature]:.2f}")
        else:
            print("❌ Не удалось проанализировать изображение")


def main():
    """Основная функция диагностики"""
    parser = argparse.ArgumentParser(description='Диагностика диабетической ретинопатии')
    parser.add_argument('--image', '-i', help='Путь к изображению для анализа')
    parser.add_argument('--mode', '-m', choices=['origin', 'goty'], default='origin',
                        help='Режим диагностики: origin (2 признака) или goty (все признаки)')
    args = parser.parse_args()

    diagnosis = RetinopathyDiagnosis()
    if not diagnosis.load_resources():
        print("Ошибка загрузки модели. Убедитесь, что вы обучили модель перед использованием.")
        sys.exit(1)

    # Устанавливаем режим из аргументов командной строки
    diagnosis.current_mode = args.mode

    # Если указан путь к изображению как аргумент
    if args.image:
        result = diagnosis.analyze_image(args.image)
        if result:
            print(f"Режим: {result['mode']}")
            print(f"Результат: {'Есть ретинопатия' if result['prediction'] == 1 else 'Нет ретинопатии'}")
            print(f"Уверенность: {result['confidence']:.2%}")

            # Анализ признаков для одиночного изображения
            if result['prediction'] == 1:
                print("\nОбнаруженные признаки ретинопатии:")
                if result['features'].get('microaneurysms_count', 0) > FEATURE_THRESHOLDS['microaneurysms_count']:
                    print(f"- Микроаневризмы: {result['features']['microaneurysms_count']}")
                if result['features'].get('exudates_area', 0) > FEATURE_THRESHOLDS['exudates_area']:
                    print(f"- Экссудаты: {result['features']['exudates_area']:.2f}")
            else:
                print("\nПризнаки здоровой сетчатки:")
                if result['features'].get('microaneurysms_count', 0) <= FEATURE_THRESHOLDS['microaneurysms_count']:
                    print(f"- Микроаневризмы: не обнаружены")
                if result['features'].get('exudates_area', 0) <= FEATURE_THRESHOLDS['exudates_area']:
                    print(f"- Экссудаты: не обнаружены")
        else:
            print("Не удалось проанализировать изображение")
            sys.exit(1)
    else:
        # Интерактивный режим
        interactive_mode(diagnosis)


if __name__ == "__main__":
    main()