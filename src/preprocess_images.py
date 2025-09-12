import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from skimage import feature, filters, morphology
from scipy import ndimage


def detect_retinopathy_features(image_path, target_size=(128, 128)):
    """
    Улучшенное обнаружение признаков ретинопатии на изображении
    """
    # Загрузка и предобработка
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)

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

    return {
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