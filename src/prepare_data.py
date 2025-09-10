import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_analyze_features(csv_path):
    """
    Загрузка и анализ важности признаков
    """
    # Загрузка данных
    df = pd.read_csv(csv_path)
    print(f"Загружено записей: {len(df)}")

    # Проверка на пропущенные значения
    print("\nПроверка пропущенных значений:")
    print(df.isnull().sum())

    # Удаление строк с пропущенными значениями
    df = df.dropna()
    print(f"Записей после удаления пропущенных значений: {len(df)}")

    # Анализ важности признаков
    X = df.drop(['filename', 'label'], axis=1, errors='ignore')
    y = df['label']

    # Обучение случайного леса для оценки важности признаков
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Важность признаков
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nВажность признаков:")
    print(feature_importance)

    # Визуализация важности признаков
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Топ-10 самых важных признаков')
    plt.tight_layout()
    plt.savefig('../results/feature_importance.png')
    plt.show()

    return df, feature_importance


def split_data(df, feature_importance, top_n=5, test_size=0.2, random_state=42):
    """
    Разделение данных с использованием только самых важных признаков
    """
    # Выбор топ-N самых важных признаков
    top_features = feature_importance.head(top_n)['feature'].tolist()
    print(f"\nИспользуем топ-{top_n} признаков: {top_features}")

    # Разделение на признаки и целевую переменную
    X = df[top_features]
    y = df['label']

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nРазделение данных:")
    print(f"Обучающая выборка: {len(X_train)} записей")
    print(f"Тестовая выборка: {len(X_test)} записей")

    # Проверка баланса классов
    print("\nРаспределение классов в обучающей выборке:")
    print(y_train.value_counts())
    print("\nРаспределение классов в тестовой выборке:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test, top_features


if __name__ == "__main__":
    # Пути к файлам
    csv_path = "../dataset/enhanced_retinopathy_features.csv"

    # Загрузка и анализ данных
    df, feature_importance = load_and_analyze_features(csv_path)

    # Разделение данных
    X_train, X_test, y_train, y_test, top_features = split_data(df, feature_importance, top_n=5)

    # Сохранение разделенных данных
    X_train.to_csv('../dataset/X_train_enhanced.csv', index=False)
    X_test.to_csv('../dataset/X_test_enhanced.csv', index=False)
    y_train.to_csv('../dataset/y_train_enhanced.csv', index=False)
    y_test.to_csv('../dataset/y_test_enhanced.csv', index=False)

    # Сохранение списка важных признаков
    pd.Series(top_features).to_csv('../dataset/top_features.csv', index=False)

    print("\nДанные успешно подготовлены и сохранены!")