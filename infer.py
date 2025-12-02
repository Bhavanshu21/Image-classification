import argparse
import os
from typing import List

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


def build_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model(checkpoint_path: str, device: torch.device) -> tuple[nn.Module, List[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


@torch.no_grad()
def predict_image(
    image_path: str,
    model: nn.Module,
    transform: transforms.Compose,
    class_names: List[str],
    device: torch.device,
) -> tuple[str, List[float]]:
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())

    return class_names[pred_idx], probs.tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with trained image classifier")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--image_size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, class_names = load_model(args.checkpoint, device)
    transform = build_transform(args.image_size)

    label, probs = predict_image(args.image, model, transform, class_names, device)
    print(f"Predicted class: {label}")
    for cls, p in sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True):
        print(f"{cls}: {p:.4f}")


if __name__ == "__main__":
    main()


