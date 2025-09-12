import os
import csv
import logging
from pathlib import Path
from config import DATASET_DIR, IMAGES_DIR

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_label_csv():
    """Создает CSV файл с метками изображений"""
    csv_path = DATASET_DIR / 'label.csv'

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'label'])

            for label in ['0', '1']:
                class_path = IMAGES_DIR / label
                if class_path.exists():
                    images_count = 0
                    for filename in os.listdir(class_path):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            writer.writerow([filename, int(label)])
                            images_count += 1
                    logger.info(f"Класс {label}: обработано {images_count} изображений")
                else:
                    logger.warning(f"Директория {class_path} не существует")

        logger.info(f"CSV-файл создан: {csv_path}")
        return True

    except Exception as e:
        logger.error(f"Ошибка при создании CSV: {e}")
        return False


if __name__ == "__main__":
    create_label_csv()