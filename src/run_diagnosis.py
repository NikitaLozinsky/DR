import argparse
import os
import sys
import cv2
import numpy as np
import pandas as pd
import joblib
from skimage import feature, filters, morphology
from sklearn.preprocessing import StandardScaler


class RetinopathyPredictor:
    def __init__(self, model_path, scaler_path, features_path):
        """
        Инициализация предсказателя с улучшенными признаками
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.top_features = pd.read_csv(features_path).iloc[:, 0].tolist()
        self.target_size = (128, 128)

    def extract_features(self, image_path):
        """
        Извлечение улучшенных признаков ретинопатии из изображения
        """
        # Загрузка и предобработка
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.target_size)

        # Нормализация гистограммы для улучшения контраста
        normalized = cv2.equalizeHist(resized)

        # 1. Базовые признаки яркости (сохраняем как требуется в ТЗ)
        mean_brightness = np.mean(resized)
        std_brightness = np.std(resized)

        # 2. Обнаружение микроАневризм (темные круглые объекты)
        _, dark_thresh = cv2.threshold(normalized, 30, 255, cv2.THRESH_BINARY_INV)
        dark_objects = morphology.remove_small_objects(dark_thresh.astype(bool), min_size=5)
        microaneurysms_count = np.sum(dark_objects)

        # 3. Обнаружение экссудатов (светлые области)
        _, light_thresh = cv2.threshold(normalized, 200, 255, cv2.THRESH_BINARY)
        light_objects = morphology.remove_small_objects(light_thresh.astype(bool), min_size=5)
        exudates_area = np.sum(light_objects)

        # 4. Признаки текстуры (LBP)
        lbp = feature.local_binary_pattern(normalized, 8, 1, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)

        # 5. Градиенты и края
        gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1)
        magnitude, _ = cv2.cartToPolar(gx, gy)
        gradient_mean = np.mean(magnitude)

        # 6. Энтропия как мера текстуры
        entropy = filters.rank.entropy(normalized, np.ones((3, 3)))
        entropy_mean = np.mean(entropy)

        # 7. Особенности сосудов (используем морфологические операции)
        blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
        _, vessel_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        vessel_skeleton = morphology.skeletonize(vessel_thresh.astype(bool))
        vessel_length = np.sum(vessel_skeleton)

        # 8. Отношение площадей различных регионов
        total_dark_area = np.sum(dark_objects)
        total_light_area = np.sum(light_objects)
        dark_to_light_ratio = total_dark_area / (total_light_area + 1e-6)

        # Создание словаря всех признаков
        features = {
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'microaneurysms_count': microaneurysms_count,
            'exudates_area': exudates_area,
            'vessel_length': vessel_length,
            'dark_to_light_ratio': dark_to_light_ratio,
            'gradient_mean': gradient_mean,
            'entropy_mean': entropy_mean,
            **{f'lbp_{i}': lbp_hist[i] for i in range(len(lbp_hist))}
        }

        # Выбор только нужных признаков (тех, что использовались при обучении)
        selected_features = {f: features[f] for f in self.top_features if f in features}

        return selected_features

    def predict(self, image_path):
        """
        Предсказание для одного изображения
        """
        # Извлечение признаков
        features = self.extract_features(image_path)

        # Создание DataFrame
        features_df = pd.DataFrame([features])

        # Масштабирование признаков
        features_scaled = self.scaler.transform(features_df)

        # Предсказание
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]

        # Интерпретация результата
        result = "Есть диабетическая ретинопатия" if prediction == 1 else "Нет диабетической ретинопатии"
        confidence = probability[1] if prediction == 1 else probability[0]

        return {
            'prediction': prediction,
            'result': result,
            'confidence': confidence,
            'features': features
        }


def clean_path(path_string):
    """
    Очистка пути от кавычек и лишних пробелов
    """
    return path_string.strip().strip('"').strip("'")


def main():
    """
    Основная функция для демонстрации работы предсказателя
    """
    # Пути к модели, scaler и признакам
    model_path = "../results/best_model_enhanced.pkl"
    scaler_path = "../results/scaler_enhanced.pkl"
    features_path = "../dataset/top_features.csv"

    # Проверка существования файлов
    for path in [model_path, scaler_path, features_path]:
        if not os.path.exists(path):
            print(f"Ошибка: файл {path} не найден!")
            print("Убедитесь, что вы обучили модель перед использованием этого скрипта.")
            return

    # Инициализация предсказателя
    predictor = RetinopathyPredictor(model_path, scaler_path, features_path)

    print("=" * 60)
    print("ДИАГНОСТИКА ДИАБЕТИЧЕСКОЙ РЕТИНОПАТИИ")
    print("=" * 60)
    print("Введите путь к изображению для анализа или 'exit' для выхода")

    # Бесконечный цикл для обработки нескольких изображений
    while True:
        # Получение пути от пользователя
        user_input = input("\nПуть к изображению: ").strip()

        # Проверка на выход
        if user_input.lower() in ['exit', 'quit', 'выход']:
            print("Завершение работы...")
            break

        # Очистка пути от кавычек
        image_path = clean_path(user_input)

        # Проверка существования файла
        if not os.path.exists(image_path):
            print(f"Ошибка: файл '{image_path}' не существует!")
            continue

        try:
            # Предсказание
            result = predictor.predict(image_path)

            # Вывод результатов
            print("\n" + "=" * 50)
            print("РЕЗУЛЬТАТ АНАЛИЗА РЕТИНОПАТИИ:")
            print("=" * 50)
            print(f"Изображение: {image_path}")
            print(f"Результат: {result['result']}")
            print(f"Уверенность: {result['confidence']:.2%}")

            # Дополнительная информация в зависимости от результата
            if result['prediction'] == 1:
                print("\nПризнаки ретинопатии:")
                if result['features'].get('microaneurysms_count', 0) > 0:
                    print(f"- Обнаружены микроаневризмы: {result['features']['microaneurysms_count']}")
                if result['features'].get('exudates_area', 0) > 0:
                    print(f"- Обнаружены экссудаты: {result['features']['exudates_area']}")
                if result['features'].get('dark_to_light_ratio', 0) > 0.5:
                    print(
                        f"- Высокое соотношение темных/светлых областей: {result['features']['dark_to_light_ratio']:.2f}")
            else:
                print("\nПризнаки здоровой сетчатки:")
                if result['features'].get('microaneurysms_count', 0) == 0:
                    print("- Микроаневризмы не обнаружены")
                if result['features'].get('exudates_area', 0) == 0:
                    print("- Экссудаты не обнаружены")

            print("=" * 50)

        except Exception as e:
            print(f"Ошибка при обработке изображения: {e}")
            print("Убедитесь, что это корректное изображение глазного дна.")


if __name__ == "__main__":
    main()