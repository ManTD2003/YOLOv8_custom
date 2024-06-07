from ultralytics import YOLO


def train():
    # Load a modeld
    model = YOLO(
        "/home/s/man/yolov8m.pt"
    )  # load a pretrained model (recommended for training)

    # Train the model
    model.train(
        data="/home/s/man/data.yaml",
        epochs=100,
        imgsz=640,
        batch=64,
        project="/home/s/man",
        name="Batch_Train",
        exist_ok=True,
        optimizer="Adam",
        lr0=1e-3,
        device="0",
    )

if __name__ == "__main__":
    train()