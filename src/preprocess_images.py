import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from skimage import feature, filters


def detect_retinopathy_features(image_path, target_size=(128, 128)):
    """
    Обнаружение признаков ретинопатии на изображении
    """
    # Загрузка и базовая предобработка
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)

    # 1. Базовые признаки яркости
    mean_brightness = np.mean(resized)
    std_brightness = np.std(resized)

    # 2. Обнаружение пятен (бинарный порог + морфологические операции)
    _, thresh = cv2.threshold(resized, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтрация контуров по размеру (игнорируем слишком маленькие)
    spots_count = 0
    total_spot_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5:  # минимальный размер пятна
            spots_count += 1
            total_spot_area += area

    # 3. Признаки текстуры (LBP - Local Binary Patterns)
    lbp = feature.local_binary_pattern(resized, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # нормализация

    # 4. Градиенты (для обнаружения резких изменений)
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1)
    magnitude, _ = cv2.cartToPolar(gx, gy)
    gradient_mean = np.mean(magnitude)

    # 5. Энтропия (мера текстуры и сложности)
    entropy = filters.rank.entropy(resized, np.ones((3, 3)))
    entropy_mean = np.mean(entropy)

    return {
        'mean_brightness': mean_brightness,
        'std_brightness': std_brightness,
        'spots_count': spots_count,
        'total_spot_area': total_spot_area,
        'gradient_mean': gradient_mean,
        'entropy_mean': entropy_mean,
        **{f'lbp_{i}': lbp_hist[i] for i in range(len(lbp_hist))}
    }


def create_enhanced_dataset(csv_path, images_base_path, output_csv_path):
    """
    Создание улучшенного датасета с признаками ретинопатии
    """
    labels_df = pd.read_csv(csv_path)

    features_list = []

    for index, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        filename = row['filename']
        label = row['label']

        # Определение пути к изображению
        image_path = os.path.join(images_base_path, str(label), filename)

        # Извлечение признаков
        features = detect_retinopathy_features(image_path)

        if features is not None:
            features['label'] = label
            features['filename'] = filename
            features_list.append(features)

    # Создание DataFrame
    df = pd.DataFrame(features_list)
    df.to_csv(output_csv_path, index=False)
    print(f"Улучшенный датасет сохранен в: {output_csv_path}")

    return df


if __name__ == "__main__":
    create_enhanced_dataset(
        "../dataset/label.csv",
        "../dataset/images",
        "../dataset/enhanced_retinopathy_features.csv"
    )