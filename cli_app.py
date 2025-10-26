import argparse
import os
import sys
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------------------------
# Config
# -------------------------
MODEL_PATH = "tomato_ensemble_best.keras"
IMG_SIZE = (224, 224)

CLASS_LABELS = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# -------------------------
# Load Model
# -------------------------
print("🔄 Loading model...")
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# -------------------------
# Preprocess Image
# -------------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# -------------------------
# Predict Image
# -------------------------
def predict_image(img_path):
    img_array = preprocess_image(img_path)

    try:
        # If model expects 2 inputs (ensemble)
        predictions = model.predict([img_array, img_array], verbose=0)
    except ValueError:
        # If model is single-input
        predictions = model.predict(img_array, verbose=0)

    if isinstance(predictions, list):
        predictions = predictions[0]

    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100
    return CLASS_LABELS[predicted_class], confidence, predictions[0]

# -------------------------
# Save CSV
# -------------------------
def save_to_csv(results, csv_file):
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Image", "Predicted_Label", "Confidence"] + CLASS_LABELS
        writer.writerow(header)

        for img_name, label, confidence, probs in results:
            row = [img_name, label, f"{confidence:.2f}%"] + [f"{p*100:.2f}%" for p in probs]
            writer.writerow(row)

    print(f"📂 Results saved to {csv_file}")

# -------------------------
# Main CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Tomato Leaf Disease Detection CLI")
    parser.add_argument("images", nargs="+", help="Path(s) to image(s) or folder(s)")
    parser.add_argument("--csv", help="Export results to CSV")
    args = parser.parse_args()

    # Collect image files
    image_paths = []
    for path in args.images:
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(path, f))
        elif os.path.isfile(path):
            image_paths.append(path)
        else:
            print(f"⚠️ Skipping invalid path: {path}")

    if not image_paths:
        print("❌ No valid images found.")
        sys.exit(1)

    results = []
    for img_path in image_paths:
        try:
            label, confidence, all_probs = predict_image(img_path)

            print(f"\n🌱 Prediction for {os.path.basename(img_path)}")
            print(f"   Disease    : {label}")
            print(f"   Confidence : {confidence:.2f}%\n")

            print("📊 Full Probabilities:")
            for cls, prob in zip(CLASS_LABELS, all_probs):
                print(f"   {cls:40s}: {prob*100:.2f}%")

            results.append((os.path.basename(img_path), label, confidence, all_probs))

        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

    # Save results to CSV if requested
    if args.csv:
        save_to_csv(results, args.csv)

if __name__ == "__main__":
    main()
