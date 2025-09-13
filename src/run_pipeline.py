import subprocess
import sys
from pathlib import Path


def run_script(script_name, description):
    """Запускает Python скрипт и возвращает результат"""
    print(f"\n{'=' * 50}")
    print(f"Запуск: {description}")
    print(f"{'=' * 50}")

    try:
        result = subprocess.run([sys.executable, script_name],
                                cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"Ошибка при запуске {script_name}: {e}")
        return False


def main():
    """Основная функция запуска пайплайна"""
    print("Запуск пайплайна обработки изображений и обучения модели")

    # Шаг 1: Настройка конфигурации (опционально)
    configure = input("Хотите настроить параметры конфигурации? (y/n): ").strip().lower()
    if configure == 'y':
        if not run_script("configure.py", "Настройка параметров"):
            print("Прерывание выполнения из-за ошибки в настройке конфигурации")
            return

    # Шаг 2: Создание CSV файла с метками
    if not run_script("create_csv.py", "Создание CSV файла с метками изображений"):
        print("Прерывание выполнения из-за ошибки при создании CSV")
        return

    # Шаг 3: Предобработка изображений
    if not run_script("preprocess_images.py", "Предобработка изображений"):
        print("Прерывание выполнения из-за ошибки при предобработке изображений")
        return

    # Шаг 4: Подготовка данных для обучения
    if not run_script("prepare_data.py", "Подготовка данных для обучения"):
        print("Прерывание выполнения из-за ошибки при подготовке данных")
        return

    # Шаг 5: Обучение модели
    if not run_script("train_model.py", "Обучение модели машинного обучения"):
        print("Прерывание выполнения из-за ошибки при обучении модели")
        return

    # Шаг 6: Запуск диагностики (опционально)
    run_diagnosis = input("Хотите запустить диагностику? (y/n): ").strip().lower()
    if run_diagnosis == 'y':
        if not run_script("run_diagnosis.py", "Запуск диагностики"):
            print("Диагностика завершилась с ошибкой")
        else:
            print("Диагностика успешно завершена")

    print("\nПайплайн завершен!")


if __name__ == "__main__":
    main()