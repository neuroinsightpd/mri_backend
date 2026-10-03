from io import BytesIO
from pathlib import Path
import json

import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite


PACKAGE_DIR = Path(__file__).resolve().parent

MODEL_PATH = PACKAGE_DIR / "mri_efficientnetb0_fine_tuned.tflite"
LABELS_PATH = PACKAGE_DIR / "labels.json"
METADATA_PATH = PACKAGE_DIR / "model_metadata.json"


class MRIModelService:
    """
    Lightweight TFLite version of the MRI prediction service. Uses the
    tiny `tflite_runtime` interpreter instead of full TensorFlow so it
    fits inside Render's free-tier 512 MB RAM limit. Same preprocessing
    and prediction logic as the original Keras-based version.
    """

    def __init__(self):
        self.interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with open(LABELS_PATH, "r", encoding="utf-8") as file:
            self.labels = json.load(file)

        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.threshold = float(self.metadata["threshold"])

    @staticmethod
    def preprocess(image_bytes: bytes):
        image = Image.open(BytesIO(image_bytes)).convert("L")
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        image = image.convert("RGB")

        image_array = np.asarray(image, dtype=np.float32)
        return np.expand_dims(image_array, axis=0)

    def predict(self, image_bytes: bytes):
        processed_image = self.preprocess(image_bytes)

        self.interpreter.set_tensor(
            self.input_details[0]["index"], processed_image
        )
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        parkinson_probability = float(output[0][0])

        predicted_class = int(parkinson_probability >= self.threshold)
        predicted_label = self.labels[str(predicted_class)]

        confidence = (
            parkinson_probability
            if predicted_class == 1
            else 1.0 - parkinson_probability
        )

        return {
            "modality": "brain_mri",
            "prediction": predicted_label,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 6),
            "probabilities": {
                "No_parkinson": round(1.0 - parkinson_probability, 6),
                "parkinson": round(parkinson_probability, 6),
            },
            "threshold": self.threshold,
            "model_name": self.metadata["model_name"],
            "model_version": self.metadata["model_version"],
            "requires_clinician_review": True,
            "disclaimer": self.metadata["disclaimer"],
        }
