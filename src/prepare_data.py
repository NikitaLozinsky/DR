import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import DATASET_DIR, RESULTS_DIR, RANDOM_STATE, TEST_SIZE, TOP_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self):
        self.feature_importance = None

    def load_and_analyze(self, csv_path):
        """Загрузка и анализ данных"""
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Загружено {len(df)} записей")

            # Проверка и очистка данных
            df = self._clean_data(df)

            # Анализ признаков
            self._analyze_features(df)
            return df

        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            raise

    def _clean_data(self, df):
        """Очистка и предобработка данных"""
        # Проверка пропущенных значений
        if df.isnull().sum().any():
            logger.warning("Обнаружены пропущенные значения")
            df = df.dropna()
        return df

    def _analyze_features(self, df):
        """Анализ важности признаков"""
        X = df.drop(['filename', 'label'], axis=1, errors='ignore')
        y = df['label']

        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        model.fit(X, y)

        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

    def split_data(self, df, top_n=TOP_FEATURES):
        """Разделение данных на train/test"""
        top_features = self.feature_importance.head(top_n)['feature'].tolist()

        X = df[top_features]
        y = df['label']

        return train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )


def main():
    """Основная функция"""
    try:
        processor = DataPreprocessor()
        df = processor.load_and_analyze(DATASET_DIR / 'enhanced_retinopathy_features.csv')

        X_train, X_test, y_train, y_test = processor.split_data(df)

        # Сохранение данных
        X_train.to_csv(DATASET_DIR / 'X_train_enhanced.csv', index=False)
        X_test.to_csv(DATASET_DIR / 'X_test_enhanced.csv', index=False)
        y_train.to_csv(DATASET_DIR / 'y_train_enhanced.csv', index=False)
        y_test.to_csv(DATASET_DIR / 'y_test_enhanced.csv', index=False)

        # Сохранение информации о признаках
        pd.Series(processor.feature_importance.head(TOP_FEATURES)['feature'].tolist()).to_csv(
            DATASET_DIR / 'top_features.csv', index=False
        )

        logger.info("Данные успешно подготовлены")

    except Exception as e:
        logger.error(f"Ошибка в подготовке данных: {e}")


if __name__ == "__main__":
    main()