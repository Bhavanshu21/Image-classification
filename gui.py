import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch
from PIL import Image, ImageTk

from infer import load_model, build_transform


class ImageClassifierGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Image Classification GUI")
        self.root.geometry("600x500")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.class_names = []
        self.transform = None
        self.image_tk = None  # keep reference so image is not garbage collected

        self._build_widgets()

    def _build_widgets(self) -> None:
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text=f"Device: {self.device}", foreground="gray").pack(
            side=tk.LEFT
        )

        load_btn = ttk.Button(top_frame, text="Load Model", command=self.load_model)
        load_btn.pack(side=tk.RIGHT)

        middle_frame = ttk.Frame(self.root, padding=10)
        middle_frame.pack(fill=tk.BOTH, expand=True)

        # Image preview
        self.image_label = ttk.Label(middle_frame, text="No image selected")
        self.image_label.pack(side=tk.LEFT, padx=10, pady=10)

        # Prediction panel
        right_frame = ttk.Frame(middle_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.status_label = ttk.Label(
            right_frame, text="Load model to start", foreground="blue"
        )
        self.status_label.pack(anchor=tk.W, pady=(0, 10))

        self.pred_label = ttk.Label(
            right_frame, text="Prediction: -", font=("Segoe UI", 12, "bold")
        )
        self.pred_label.pack(anchor=tk.W, pady=(0, 10))

        self.probs_text = tk.Text(right_frame, height=10, width=40, state=tk.DISABLED)
        self.probs_text.pack(fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(self.root, padding=10)
        bottom_frame.pack(fill=tk.X)

        self.select_btn = ttk.Button(
            bottom_frame, text="Select Image", command=self.select_image, state=tk.DISABLED
        )
        self.select_btn.pack(side=tk.LEFT)

    def load_model(self) -> None:
        """Load model checkpoint from outputs/best_model.pth."""

        default_path = os.path.join("outputs", "best_model.pth")
        if not os.path.exists(default_path):
            messagebox.showerror(
                "Model not found",
                f"Could not find model checkpoint at:\n{default_path}\n\n"
                "Train the model first using train.py.",
            )
            return

        self.status_label.config(text="Loading model...", foreground="blue")
        self.root.update_idletasks()

        def _load():
            try:
                model, class_names = load_model(default_path, self.device)
                self.model = model
                self.class_names = class_names
                self.transform = build_transform()

                self.status_label.config(text="Model loaded. Select an image.", foreground="green")
                self.select_btn.config(state=tk.NORMAL)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Error", f"Failed to load model:\n{exc}")
                self.status_label.config(text="Failed to load model", foreground="red")

        threading.Thread(target=_load, daemon=True).start()

    def select_image(self) -> None:
        if self.model is None or self.transform is None:
            messagebox.showwarning("Model not loaded", "Please load the model first.")
            return

        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select image", filetypes=filetypes)
        if not path:
            return

        self._show_image(path)
        self._run_prediction(path)

    def _show_image(self, path: str) -> None:
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((256, 256))
            self.image_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.image_tk, text="")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Failed to open image:\n{exc}")

    def _run_prediction(self, path: str) -> None:
        self.status_label.config(text="Running prediction...", foreground="blue")
        self.pred_label.config(text="Prediction: -")
        self._set_probs_text("...")
        self.root.update_idletasks()

        def _predict():
            try:
                img = Image.open(path).convert("RGB")
                tensor = self.transform(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = self.model(tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

                pred_idx = int(probs.argmax())
                pred_label = self.class_names[pred_idx]

                # Update UI from main thread
                def _update_ui():
                    self.pred_label.config(text=f"Prediction: {pred_label}")
                    self.status_label.config(text="Done", foreground="green")

                    lines = [
                        f"{cls}: {p:.4f}"
                        for cls, p in sorted(
                            zip(self.class_names, probs), key=lambda x: x[1], reverse=True
                        )
                    ]
                    self._set_probs_text("\n".join(lines))

                self.root.after(0, _update_ui)
            except Exception as exc:  # noqa: BLE001
                def _on_error():
                    messagebox.showerror("Error", f"Failed to run prediction:\n{exc}")
                    self.status_label.config(text="Prediction failed", foreground="red")

                self.root.after(0, _on_error)

        threading.Thread(target=_predict, daemon=True).start()

    def _set_probs_text(self, text: str) -> None:
        self.probs_text.config(state=tk.NORMAL)
        self.probs_text.delete("1.0", tk.END)
        self.probs_text.insert(tk.END, text)
        self.probs_text.config(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    app = ImageClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()


