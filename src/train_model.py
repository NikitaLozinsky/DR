import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def load_enhanced_data():
    """
    Загрузка улучшенных данных
    """
    X_train = pd.read_csv('../dataset/X_train_enhanced.csv')
    X_test = pd.read_csv('../dataset/X_test_enhanced.csv')
    y_train = pd.read_csv('../dataset/y_train_enhanced.csv').values.ravel()
    y_test = pd.read_csv('../dataset/y_test_enhanced.csv').values.ravel()

    # Загрузка списка важных признаков
    top_features = pd.read_csv('../dataset/top_features.csv').iloc[:, 0].tolist()

    return X_train, X_test, y_train, y_test, top_features


def normalize_data(X_train, X_test):
    """
    Нормализация данных
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Сохраняем scaler для будущих предсказаний
    joblib.dump(scaler, '../results/scaler_enhanced.pkl')

    return X_train_scaled, X_test_scaled, scaler


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Обучение и оценка нескольких моделей
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True)
    }

    results = {}

    for name, model in models.items():
        print(f"\nОбучение {name}...")
        model.fit(X_train, y_train)

        # Предсказания
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Оценка
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'confusion_matrix': cm
        }

        print(f"{name} Accuracy: {accuracy:.4f}")

        # ROC-AUC если доступны вероятности
        if y_prob is not None:
            roc_auc = roc_auc_score(y_test, y_prob)
            results[name]['roc_auc'] = roc_auc
            print(f"{name} ROC-AUC: {roc_auc:.4f}")

    return results


def select_and_save_best_model(results):
    """
    Выбор и сохранение лучшей модели
    """
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']

    print(f"\nЛучшая модель: {best_model_name} с точностью {best_accuracy:.4f}")

    # Сохранение лучшей модели
    joblib.dump(best_model, '../results/best_model_enhanced.pkl')
    print("Лучшая модель сохранена в '../results/best_model_enhanced.pkl'")

    # Визуализация результатов
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=accuracies, y=model_names)
    plt.title('Сравнение точности моделей')
    plt.xlabel('Accuracy')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig('../results/models_comparison.png')
    plt.show()

    return best_model_name, best_model


if __name__ == "__main__":
    # Загрузка данных
    X_train, X_test, y_train, y_test, top_features = load_enhanced_data()

    # Нормализация данных
    X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)

    # Обучение и оценка моделей
    print("Обучение и оценка моделей...")
    results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Выбор и сохранение лучшей модели
    best_model_name, best_model = select_and_save_best_model(results)

    # Детальный отчет для лучшей модели
    best_result = results[best_model_name]
    y_pred = best_model.predict(X_test_scaled)

    print(f"\nДетальный отчет для {best_model_name}:")
    print("Confusion Matrix:")
    print(best_result['confusion_matrix'])
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Визуализация матрицы ошибок
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['No DR', 'DR'],
                yticklabels=['No DR', 'DR'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.savefig('../results/confusion_matrix_enhanced.png')
    plt.show()