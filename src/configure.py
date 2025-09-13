import json
from pathlib import Path
from config import load_config, CONFIG_FILE, DEFAULT_CONFIG


def load_config():
    """Загружает конфигурацию из файла или использует значения по умолчанию"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_CONFIG


def save_config(config):
    """Сохраняет конфигурацию в файл с относительными путями"""
    base_dir = Path(__file__).parent
    config_to_save = config.copy()

    # Преобразуем абсолютные пути в относительные
    for key in ['DATASET_DIR', 'RESULTS_DIR', 'IMAGES_DIR']:
        if key in config_to_save:
            path = Path(config_to_save[key])
            try:
                # Пытаемся получить относительный путь
                relative_path = path.relative_to(base_dir)
                config_to_save[key] = str(relative_path)
            except ValueError:
                # Если путь не внутри базовой директории, оставляем как есть
                pass

    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    print("Конфигурация сохранена!")

def show_current_config(config):
    """Показывает текущую конфигурацию"""
    print("\nТекущая конфигурация:")
    for key, value in config.items():
        print(f"  {key}: {value}")


def configure_interactive():
    """Интерактивная настройка конфигурации"""
    config = load_config()

    while True:
        show_current_config(config)
        print("\nЧто вы хотите изменить?")
        print("1. IMAGE_SIZE (размер изображений)")
        print("2. THRESHOLDS (пороговые значения)")
        print("3. DATASET_DIR (папка датасета)")
        print("4. RESULTS_DIR (папка результатов)")
        print("5. IMAGES_DIR (папка изображений)")
        print("6. Сбросить всё на значения по умолчанию")
        print("7. Сохранить и выйти")
        print("8. Выйти без сохранения")

        choice = input("\nВыберите вариант (1-8): ").strip()

        if choice == '1':
            try:
                width = int(input("Ширина изображения: "))
                height = int(input("Высота изображения: "))
                config['IMAGE_SIZE'] = [width, height]
            except ValueError:
                print("Ошибка: введите числа!")

        elif choice == '2':
            try:
                dark = int(input("Порог для 'dark': "))
                light = int(input("Порог для 'light': "))
                config['THRESHOLDS'] = {'dark': dark, 'light': light}
            except ValueError:
                print("Ошибка: введите числа!")

        elif choice == '3':
            new_path = input("Новый путь к DATASET_DIR: ").strip()
            config['DATASET_DIR'] = new_path

        elif choice == '4':
            new_path = input("Новый путь к RESULTS_DIR: ").strip()
            config['RESULTS_DIR'] = new_path

        elif choice == '5':
            new_path = input("Новый путь к IMAGES_DIR: ").strip()
            config['IMAGES_DIR'] = new_path

        elif choice == '6':
            config = DEFAULT_CONFIG.copy()
            print("Конфигурация сброшена к значениям по умолчанию!")

        elif choice == '7':
            save_config(config)
            break

        elif choice == '8':
            print("Выход без сохранения")
            break

        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    configure_interactive()