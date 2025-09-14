import pandas as pd
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from pathlib import Path
from config import DATASET_DIR, RESULTS_DIR, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def load_data(self):
        """Загрузка подготовленных данных"""
        try:
            X_train = pd.read_csv(DATASET_DIR / 'X_train_enhanced.csv')
            X_test = pd.read_csv(DATASET_DIR / 'X_test_enhanced.csv')
            y_train = pd.read_csv(DATASET_DIR / 'y_train_enhanced.csv').values.ravel()
            y_test = pd.read_csv(DATASET_DIR / 'y_test_enhanced.csv').values.ravel()

            logger.info(f"Загружены данные: X_train={X_train.shape}, X_test={X_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            raise

    def train_models(self, X_train, X_test, y_train, y_test):
        """Обучение и оценка различных моделей с подбором гиперпараметров"""
        models = {
            'Logistic Regression': (LogisticRegression(max_iter=10000, random_state=RANDOM_STATE), {
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            }),
            'Random Forest': (RandomForestClassifier(random_state=RANDOM_STATE), {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }),
            'Gradient Boosting': (GradientBoostingClassifier(random_state=RANDOM_STATE), {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7],
                'classifier__subsample': [0.8, 0.9, 1.0]
            }),
            'SVM': (SVC(probability=True, random_state=RANDOM_STATE), {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale', 'auto']
            }),
            'K-Nearest Neighbors': (KNeighborsClassifier(), {
                'classifier__n_neighbors': [3, 5, 7, 9, 11],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__p': [1, 2]  # 1: Manhattan, 2: Euclidean
            })
        }

        results = {}

        for name, (model, params) in models.items():
            try:
                logger.info(f"Обучение {name}...")

                # Создаем пайплайн с масштабированием и моделью
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', model)
                ])

                # Если это логистическая регрессия с penalty='l1', нужен solver, который поддерживает l1
                if name == 'Logistic Regression':
                    # Убедимся, что для l1 penalty используется подходящий solver
                    param_grid = []
                    for penalty in params['classifier__penalty']:
                        if penalty == 'l1':
                            param_set = {k: v for k, v in params.items()}
                            param_set['classifier__solver'] = ['liblinear', 'saga']
                            param_set['classifier__penalty'] = [penalty]
                            param_grid.append(param_set)
                        else:
                            param_set = {k: v for k, v in params.items()}
                            param_set['classifier__solver'] = ['liblinear', 'saga', 'newton-cg', 'lbfgs']
                            param_set['classifier__penalty'] = [penalty]
                            param_grid.append(param_set)
                else:
                    param_grid = params

                start_time = time()
                grid_search = GridSearchCV(
                    pipeline, param_grid,
                    cv=self.cv,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1
                )

                grid_search.fit(X_train, y_train)
                training_time = time() - start_time

                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]

                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)

                results[name] = {
                    'model': best_model,
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'best_params': grid_search.best_params_,
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'training_time': training_time,
                    'feature_importance': self._get_feature_importance(best_model, X_train.columns)
                }

                logger.info(f"{name} - Лучшие параметры: {grid_search.best_params_}")
                logger.info(
                    f"{name} - Точность: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}, Время обучения: {training_time:.2f}с")

                # Сохраняем отчет по классификации
                report = classification_report(y_test, y_pred, output_dict=True)
                pd.DataFrame(report).transpose().to_csv(RESULTS_DIR / f'{name.lower().replace(" ", "_")}_report.csv')

            except Exception as e:
                logger.error(f"Ошибка обучения {name}: {e}")
                import traceback
                traceback.print_exc()

        return results

    def _get_feature_importance(self, model, feature_names):
        """Извлечение важности признаков из модели"""
        try:
            if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                return dict(zip(feature_names, model.named_steps['classifier'].feature_importances_))
            elif hasattr(model.named_steps['classifier'], 'coef_'):
                # Для линейных моделей берем абсолютные значения коэффициентов
                return dict(zip(feature_names, np.abs(model.named_steps['classifier'].coef_[0])))
            else:
                return {}
        except:
            return {}

    def save_results(self, results):
        """Сохранение результатов и моделей"""
        # Создаем директорию для результатов, если не существует
        RESULTS_DIR.mkdir(exist_ok=True)

        # Сохранение лучшей модели
        self.best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_model = results[self.best_model_name]['model']

        joblib.dump(self.best_model, RESULTS_DIR / 'best_model_enhanced.pkl')
        joblib.dump(self.scaler, RESULTS_DIR / 'scaler_enhanced.pkl')

        # Сохранение всех результатов
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[m]['accuracy'] for m in results],
            'ROC-AUC': [results[m]['roc_auc'] for m in results],
            'Training Time (s)': [results[m]['training_time'] for m in results]
        })
        results_df.to_csv(RESULTS_DIR / 'model_comparison.csv', index=False)

        # Сохранение лучших параметров для каждой модели
        best_params_df = pd.DataFrame(
            [(name, results[name]['best_params']) for name in results],
            columns=['Model', 'Best Parameters']
        )
        best_params_df.to_csv(RESULTS_DIR / 'best_parameters.csv', index=False)

        # Визуализация результатов
        self._create_plots(results)

        logger.info(
            f"Лучшая модель: {self.best_model_name} с точностью {results[self.best_model_name]['accuracy']:.4f}")

    def _create_plots(self, results):
        """Создание графиков результатов"""
        try:
            # 1. Сравнение точности моделей
            plt.figure(figsize=(10, 6))
            models = list(results.keys())
            accuracies = [results[m]['accuracy'] for m in models]
            plt.bar(models, accuracies)
            plt.title('Сравнение точности моделей')
            plt.xlabel('Модель')
            plt.ylabel('Точность')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / 'model_accuracy_comparison.png')
            plt.close()

            # 2. Матрица ошибок для лучшей модели
            plt.figure(figsize=(8, 6))
            cm = results[self.best_model_name]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Матрица ошибок: {self.best_model_name}')
            plt.ylabel('Истинный класс')
            plt.xlabel('Предсказанный класс')
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / 'best_model_confusion_matrix.png')
            plt.close()

            # 3. ROC-кривая для лучшей модели
            plt.figure(figsize=(8, 6))
            X_test = pd.read_csv(DATASET_DIR / 'X_test_enhanced.csv')
            y_test = pd.read_csv(DATASET_DIR / 'y_test_enhanced.csv').values.ravel()

            y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
            RocCurveDisplay.from_predictions(y_test, y_pred_proba)
            plt.title(f'ROC-кривая: {self.best_model_name}')
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / 'best_model_roc_curve.png')
            plt.close()

            # 4. Важность признаков для моделей, где это применимо
            for model_name, result in results.items():
                if result['feature_importance']:
                    importance_dict = result['feature_importance']
                    if importance_dict:  # Проверяем, не пустой ли словарь
                        # Сортируем признаки по важности
                        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                        features, importance = zip(*sorted_importance)

                        plt.figure(figsize=(10, 6))
                        plt.barh(features, importance)
                        plt.title(f'Топ-10 важных признаков: {model_name}')
                        plt.xlabel('Важность')
                        plt.tight_layout()
                        plt.savefig(RESULTS_DIR / f'{model_name.lower().replace(" ", "_")}_feature_importance.png')
                        plt.close()

        except Exception as e:
            logger.error(f"Ошибка при создании графиков: {e}")


def main():
    """Основная функция обучения"""
    try:
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.load_data()

        results = trainer.train_models(X_train, X_test, y_train, y_test)
        trainer.save_results(results)

        logger.info("Обучение завершено успешно!")

    except Exception as e:
        logger.error(f"Ошибка в процессе обучения: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()