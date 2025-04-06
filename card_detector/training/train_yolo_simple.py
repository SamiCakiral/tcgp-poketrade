from ultralytics import YOLO
import torch
import os

# Configuration simplifiée
MODEL_SIZE = "n"  # nano est suffisant pour 2 classes
EPOCHS = 100
BATCH_SIZE = 2
IMAGE_SIZE = 640
DEVICE = "mps"  # ou "cpu" si MPS pose problème

def create_simple_yaml():
    """Crée un fichier YAML simplifié avec 2 classes"""
    yaml_content = {
        "path": os.path.abspath("IA/yolo_generated_dataset"),
        "train": "images/",
        "val": "validation/images/",
        "nc": 2,  # Seulement 2 classes
        "names": {
            0: "empty_slot",
            1: "card"  # Toute carte est simplement une "card"
        }
    }
    
    import yaml
    yaml_path = os.path.join("IA/yolo_generated_dataset", "pokemon_cards_simple.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

def train_detector():
    """Entraîne le détecteur simple (vide/non-vide)"""
    model = YOLO(f"yolov8{MODEL_SIZE}.pt")
    
    results = model.train(
        data="IA/yolo_generated_dataset/pokemon_cards_simple.yaml",
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        pretrained=True,
        optimizer="SGD",
        lr0=0.001,
        patience=20,
        save_period=5,
        name="pokemon_card_detector"
    )
    
    return results 