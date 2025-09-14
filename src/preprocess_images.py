import cv2
import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm
from pathlib import Path
from config import DATASET_DIR, RESULTS_DIR, IMAGES_DIR, IMAGE_SIZE, THRESHOLDS

# Правильный импорт для современных версий scikit-image
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    # Для старых версий (до 0.19)
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

from skimage import filters, morphology, measure, feature
from scipy.stats import entropy

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
                logger.warning(f"Не удалось загрузить изображение: {image_path}")
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
        """Извлечение расширенных признаков из обработанного изображения"""
        features = {}

        # 1. Базовые статистики
        features['mean_brightness'] = np.mean(original)
        features['std_brightness'] = np.std(original)

        # 2. Важнейшие признаки: Текстура с помощью GLCM
        # Убедимся, что normalized имеет целочисленные значения
        normalized_int = normalized.astype(np.uint8)
        glcm = graycomatrix(normalized_int,
                            distances=[1, 3],
                            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            levels=256,
                            symmetric=True,
                            normed=True)

        # Извлекаем свойства из GLCM
        features['glcm_contrast'] = graycoprops(glcm, 'contrast').mean()
        features['glcm_homogeneity'] = graycoprops(glcm, 'homogeneity').mean()
        features['glcm_energy'] = graycoprops(glcm, 'energy').mean()
        features['glcm_correlation'] = graycoprops(glcm, 'correlation').mean()

        # 3. Признаки на основе градиентов (HOG)
        try:
            hog_features, _ = feature.hog(normalized,
                                          pixels_per_cell=(16, 16),
                                          cells_per_block=(2, 2),
                                          visualize=True)
            features['hog_mean'] = np.mean(hog_features)
            features['hog_std'] = np.std(hog_features)
        except Exception as e:
            logger.warning(f"HOG features not available: {e}")
            features['hog_mean'] = 0
            features['hog_std'] = 0

        # 4. Детекция специфических поражений
        # --- Детекция КАНДИДАТОВ на микроаневризмы/геморрагии (темные круглые объекты) ---
        otsu_threshold = filters.threshold_otsu(normalized)
        dark_objects = normalized < otsu_threshold * 0.4
        dark_objects_cleaned = morphology.closing(dark_objects, morphology.disk(1))
        dark_objects_cleaned = morphology.opening(dark_objects_cleaned, morphology.disk(1))
        labeled_dark = measure.label(dark_objects_cleaned)
        regions = measure.regionprops(labeled_dark)
        microaneurysm_candidates = [
            region for region in regions
            if 2 < region.area < 50 and region.eccentricity < 0.8
        ]
        features['dark_objects_count'] = len(microaneurysm_candidates)
        features['dark_objects_total_area'] = sum(
            [r.area for r in microaneurysm_candidates]) if microaneurysm_candidates else 0

        # --- Детекция КАНДИДАТОВ на экссудаты (светлые яркие объекты) ---
        # Ищем очень яркие области
        bright_objects = normalized > otsu_threshold * 1.4
        bright_objects_cleaned = morphology.closing(bright_objects, morphology.disk(1))
        bright_objects_cleaned = morphology.opening(bright_objects_cleaned, morphology.disk(1))
        labeled_bright = measure.label(bright_objects_cleaned)

        # Передаем intensity_image для вычисления свойств интенсивности
        regions_bright = measure.regionprops(labeled_bright, intensity_image=normalized)

        # Фильтруем по площади и интенсивности
        exudate_candidates = [
            region for region in regions_bright
            if 10 < region.area < 500 and region.mean_intensity > np.mean(normalized) * 1.2
        ]

        features['bright_objects_count'] = len(exudate_candidates)
        features['bright_objects_total_area'] = sum([r.area for r in exudate_candidates]) if exudate_candidates else 0
        features['bright_objects_mean_intensity'] = np.mean(
            [r.mean_intensity for r in exudate_candidates]) if exudate_candidates else 0

        # 5. Гистограмма ориентаций градиентов
        sobel_x = filters.sobel_v(normalized)
        sobel_y = filters.sobel_h(normalized)
        orientation_histogram, _ = np.histogram(np.arctan2(sobel_y, sobel_x).flatten(), bins=8, range=(-np.pi, np.pi))
        features['gradient_orientation_entropy'] = entropy(orientation_histogram)

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
            else:
                logger.warning(f"Изображение не найдено: {image_path}")

        df = pd.DataFrame(features_list)
        df.to_csv(output_path, index=False)
        logger.info(f"Создан датасет с {len(df)} записями")
        return True

    except Exception as e:
        logger.error(f"Ошибка создания датасета: {e}")
        return False


if __name__ == "__main__":
    create_enhanced_dataset()