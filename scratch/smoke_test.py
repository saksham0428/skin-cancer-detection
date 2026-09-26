import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_model, predict_image
from app.core.config import settings

def main():
    print(f"Loading model from {settings.model_path}")
    model = load_model(settings.model_path)
    
    # Pick a few sample images from the test set or anywhere
    import pandas as pd
    from src.config import PROCESSED_DIR, IMAGE_DIR
    
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    sample_images = test_df.sample(5, random_state=42)['image_id'].tolist()
    
    for img_id in sample_images:
        img_path = IMAGE_DIR / f"{img_id}.jpg"
        print(f"\nRunning inference on {img_path.name}")
        result = predict_image(img_path, model)
        print(f"Predicted Class: {result.predicted_class} ({result.predicted_class_full_name})")
        print(f"Confidence: {result.confidence}")
        
if __name__ == "__main__":
    main()
