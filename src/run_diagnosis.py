import argparse
import os
import sys
import cv2
import numpy as np
import pandas as pd
import joblib
from skimage import feature, filters
from sklearn.preprocessing import StandardScaler


class RetinopathyPredictor:
    def __init__(self, model_path, scaler_path, features_path):
        """
        Инициализация предсказателя
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.top_features = pd.read_csv(features_path).iloc[:, 0].tolist()
        self.target_size = (128, 128)

    def extract_features(self, image_path):
        """
        Извлечение признаков ретинопатии из изображения
        """
        # [Ваш код из extract_features функции]
        # Загрузка и базовая предобработка
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.target_size)

        # 1. Базовые признаки яркости
        mean_brightness = np.mean(resized)
        std_brightness = np.std(resized)

        # 2. Обнаружение пятен
        _, thresh = cv2.threshold(resized, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        spots_count = 0
        total_spot_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5:
                spots_count += 1
                total_spot_area += area

        # 3. Признаки текстуры (LBP)
        lbp = feature.local_binary_pattern(resized, 8, 1, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)

        # 4. Градиенты
        gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1)
        magnitude, _ = cv2.cartToPolar(gx, gy)
        gradient_mean = np.mean(magnitude)

        # 5. Энтропия
        entropy = filters.rank.entropy(resized, np.ones((3, 3)))
        entropy_mean = np.mean(entropy)

        # Создание словаря признаков
        features = {
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'spots_count': spots_count,
            'total_spot_area': total_spot_area,
            'gradient_mean': gradient_mean,
            'entropy_mean': entropy_mean,
            **{f'lbp_{i}': lbp_hist[i] for i in range(len(lbp_hist))}
        }

        # Выбор только нужных признаков
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


def main():
    """
    Основная функция для демонстрации работы предсказателя
    """
    # Пути к модели, scaler и признакам
    model_path = "../results/best_model_enhanced.pkl"
    scaler_path = "../results/scaler_enhanced.pkl"
    features_path = "../dataset/top_features.csv"

    # Инициализация предсказателя
    predictor = RetinopathyPredictor(model_path, scaler_path, features_path)

    # Пример использования
    image_path = input("Введите путь к изображению для анализа: ")

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
        print("\nИзвлеченные признаки:")
        for feature, value in result['features'].items():
            print(f"{feature}: {value:.4f}")
        print("=" * 50)

    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")


if __name__ == "__main__":
    main()