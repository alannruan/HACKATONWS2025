# HACKATONWS2025
!find /content/epi_dataset -name "*.yaml"
%cd /content/epi_dataset
!sed -i 's|/content/epi_dataset/train/images|/content/epi_dataset/EPI-2/train/images|g' data.yaml
!sed -i 's|/content/epi_dataset/valid/images|/content/epi_dataset/EPI-2/valid/images|g' data.yaml

!find /content/epi_dataset -type d

!pip install ultralytics --quiet

import zipfile
import os

zip_path = "My First Project.v1i.yolov5pytorch.zip"
extract_path = "/content/epi_dataset"

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Lista todos os arquivos no zip
    all_files = zip_ref.namelist()
    
    # Filtra só os que começam com a pasta desejada
    valid_files = [f for f in all_files if f.startswith("My First Project.v1i.yolov5pytorch/valid/")]
    
    # Extrai só os arquivos filtrados
    for file in valid_files:
        zip_ref.extract(file, extract_path)

print("✅ Pasta 'valid' extraída com sucesso!")


from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="/content/epi_dataset/data.yaml", epochs=20)


import cv2
import matplotlib.pyplot as plt

# Carregar o modelo treinado
model = YOLO("/content/runs/detect/epi_model/weights/best.pt")
model.names = {0: "capacete", 1: "luva"}

# Nome da imagem enviada no upload
img_path = list(uploaded.keys())[0]

# Fazer a predição
results = model(img_path, conf=0.4)

# Mostrar a imagem com as detecções
img = results[0].plot()
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 8))
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Detecção de EPIs")
plt.show()

# Mostrar no terminal as classes detectadas
for box in results[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    print(f"Detectado: {model.names[cls]} com confiança de {conf:.2f}")
