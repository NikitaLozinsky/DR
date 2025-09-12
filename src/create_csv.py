import os
import csv

# пока не трогай всё, что больше 1
base_path = r"C:\Users\decop\PycharmProjects\Project0\dataset\images"
csv_path = r"C:\Users\decop\PycharmProjects\Project0\dataset\label.csv"

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'label'])


    class0_path = os.path.join(base_path, '0')
    if os.path.exists(class0_path):
        for filename in os.listdir(class0_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                writer.writerow([filename, 0])


    class1_path = os.path.join(base_path, '1')
    if os.path.exists(class1_path):
        for filename in os.listdir(class1_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                writer.writerow([filename, 1])

print(f"CSV-файл создан: {csv_path}")