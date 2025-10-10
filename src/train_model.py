import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import DATASET_DIR, RESULTS_DIR, IMAGE_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.best_model = None

    def load_data(self):
        """Загрузка подготовленных данных"""
        try:
            X_train = pd.read_csv(DATASET_DIR / 'X_train_enhanced.csv')
            X_test = pd.read_csv(DATASET_DIR / 'X_test_enhanced.csv')
            y_train = pd.read_csv(DATASET_DIR / 'y_train_enhanced.csv').values.ravel()
            y_test = pd.read_csv(DATASET_DIR / 'y_test_enhanced.csv').values.ravel()

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            raise

    def train_origin_model(self, X_train, X_test, y_train, y_test):
        """Обучение модели только на 2 признаках из ТЗ"""
        try:
            # Отбираем только 2 базовых признака
            origin_features = ['mean_brightness', 'std_brightness']
            X_train_origin = X_train[origin_features]
            X_test_origin = X_test[origin_features]

            # Обучаем логистическую регрессию
            model = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear', random_state=42)
            model.fit(X_train_origin, y_train)

            # Предсказания и метрики
            y_pred = model.predict(X_test_origin)
            accuracy = accuracy_score(y_test, y_pred)

            logger.info(f"Origin модель (2 признака) - Accuracy: {accuracy:.4f}")
            return model, accuracy

        except Exception as e:
            logger.error(f"Ошибка обучения origin модели: {e}")
            return None, 0

    def train_models(self, X_train, X_test, y_train, y_test):
        """Обучение и оценка моделей"""
        models = {
            'Logistic Regression': (LogisticRegression(max_iter=1000), {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'saga']
            }),
            'Random Forest': (RandomForestClassifier(), {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20]
            }),
            'Gradient Boosting': (GradientBoostingClassifier(), {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 0.2]
            })
        }

        results = {}
        for name, (model, params) in models.items():
            try:
                logger.info(f"Обучение {name}...")
                grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)

                results[name] = {
                    'model': best_model,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'best_params': grid_search.best_params_,
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }

            except Exception as e:
                logger.error(f"Ошибка обучения {name}: {e}")

        return results

    def save_results(self, results):
        """Сохранение результатов и моделей"""
        # Сохранение лучшей модели
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_model = results[best_model_name]['model']

        joblib.dump(self.best_model, RESULTS_DIR / 'best_model_enhanced.pkl')
        joblib.dump(self.scaler, RESULTS_DIR / 'scaler_enhanced.pkl')

        # Визуализация результатов
        self._create_plots(results, best_model_name)

    def _create_plots(self, results, best_model_name):
        """Создание графиков результатов"""
        # [Реализация визуализации]
        pass


def main():
    """Основная функция обучения"""
    try:
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.load_data()

        # Масштабирование признаков для goty модели
        X_train_scaled = trainer.scaler.fit_transform(X_train)
        X_test_scaled = trainer.scaler.transform(X_test)

        # Обучение origin модели (только 2 признака)
        origin_model, origin_accuracy = trainer.train_origin_model(X_train, X_test, y_train, y_test)
        if origin_model:
            joblib.dump(origin_model, RESULTS_DIR / 'origin_model.pkl')
            # Сохраняем accuracy origin модели
            with open(RESULTS_DIR / 'origin_accuracy.txt', 'w') as f:
                f.write(f"Origin Model Accuracy: {origin_accuracy:.4f}")
            logger.info(f"Origin модель сохранена с accuracy: {origin_accuracy:.4f}")

        # Обучение goty моделей (все признаки)
        results = trainer.train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        trainer.save_results(results)

        logger.info("Обучение завершено успешно")

    except Exception as e:
        logger.error(f"Ошибка в процессе обучения: {e}")


if __name__ == "__main__":
    main()