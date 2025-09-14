import cv2
import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm
from skimage import feature, filters, morphology
from pathlib import Path
from config import DATASET_DIR, RESULTS_DIR, IMAGES_DIR, IMAGE_SIZE, THRESHOLDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self, target_size=IMAGE_SIZE):
        self.target_size = target_size

    def process_image(self, image_path):
        """Обработка одного изображения и извлечение признаков"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None

            # Конвейер обработки изображения
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, self.target_size)
            normalized = cv2.equalizeHist(resized)

            # Извлечение признаков
            features = self._extract_features(normalized, resized)
            return features

        except Exception as e:
            logger.error(f"Ошибка обработки {image_path}: {e}")
            return None

    def _extract_features(self, normalized, original):
        """Извлечение признаков из обработанного изображения"""
        # 1. Базовые признаки яркости
        mean_brightness = np.mean(original)
        std_brightness = np.std(original)

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

        features = {
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'microaneurysms_count': microaneurysms_count,
            'exudates_area': exudates_area,
            'vessel_length': vessel_length,
            'dark_to_light_ratio': dark_to_light_ratio,
            'gradient_mean': gradient_mean,
            'entropy_mean': entropy_mean,
        }

        # Добавляем LBP признаки
        for i in range(len(lbp_hist)):
            features[f'lbp_{i}'] = lbp_hist[i]

        return features


def create_enhanced_dataset():
    """Создание датасета с признаками"""
    csv_path = DATASET_DIR / 'label.csv'
    output_path = DATASET_DIR / 'enhanced_retinopathy_features.csv'

    try:
        labels_df = pd.read_csv(csv_path)
        processor = ImageProcessor()

        features_list = []
        for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
            filename = row['filename']
            label = row['label']
            image_path = IMAGES_DIR / str(label) / filename

            if image_path.exists():
                features = processor.process_image(image_path)
                if features:
                    features['label'] = label
                    features['filename'] = filename
                    features_list.append(features)

        df = pd.DataFrame(features_list)
        df.to_csv(output_path, index=False)
        logger.info(f"Создан датасет с {len(df)} записями")
        return True

    except Exception as e:
        logger.error(f"Ошибка создания датасета: {e}")
        return False


if __name__ == "__main__":
    create_enhanced_dataset()