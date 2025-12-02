## High-End Image Classification Model (PyTorch)

This project provides a **full working image classification pipeline** using a **pretrained ResNet‑50** backbone (state‑of‑the‑art baseline) with modern training tricks.

### 1. Install dependencies

From the project folder (`project 2`):

```bash
pip install -r requirements.txt
```

On Windows, make sure you have a Python environment where PyTorch can install (see the official PyTorch site if you need a specific wheel).

### 2. Use built-in CIFAR‑10 dataset (recommended to start)

You don’t need to download anything manually. The script will automatically download **CIFAR‑10** (10 classes, 60k images) into a `data/` folder and train on it.

### 3. Train the model (CIFAR‑10, zero setup)

Run this from the project folder:

```bash
python train.py --dataset cifar10 --epochs 20 --batch_size 64
```

You can change epochs, batch size, etc., but this command alone is enough to get a full training run with a strong model.

Useful flags:

- `--dataset cifar10` – use CIFAR‑10 (auto‑download, default).
- `--data_dir data` – where CIFAR‑10 is stored.
- `--image_size 224` – change resolution.
- `--lr 1e-3` – learning rate.
- `--no_pretrained` – turn off ImageNet pretraining (not recommended).
- `--output_dir outputs` – where checkpoints are saved.

### 4. (Optional) Use your own folder dataset instead

If later you want your own data, prepare this structure:

```text
your_dataset/
  train/
    class1/
    class2/
  val/
    class1/
    class2/
```

Then run:

```bash
python train.py --dataset folder --data_dir path\to\your_dataset --epochs 20 --batch_size 32
```

After training, the best checkpoint is stored as:

```text
outputs/best_model.pth
```

### 5. Run inference on a single image

```bash
python infer.py --checkpoint outputs\best_model.pth --image path\to\image.jpg
```

You will see the predicted class and class probabilities printed to the console.

### 6. Notes on performance

- Uses **ResNet‑50 with ImageNet weights**, AdamW optimizer, cosine LR schedule, and standard augmentations (flip, rotation, color jitter, normalization).
- For best speed/accuracy, enable a GPU if available; the script will automatically detect and use CUDA.


