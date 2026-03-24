from ultralytics import YOLO
import numpy as np
import os
from dotenv import load_dotenv
from app.utils import image_to_base64, get_confidence_color

load_dotenv()

# Class names must match your data.yaml exactly
CLASS_NAMES = ['1B-1L', '1B-1L-F', '1B-2L', '1B-2L-F', '1B-3L', 'banjhi', 'other']

# Human readable descriptions shown to the user
CLASS_DESCRIPTIONS = {
    '1B-1L'  : '1 bud + 1 leaf — premium grade',
    '1B-1L-F': '1 bud + 1 leaf with fish leaf',
    '1B-2L'  : '1 bud + 2 leaves — standard fine pluck',
    '1B-2L-F': '1 bud + 2 leaves with fish leaf',
    '1B-3L'  : '1 bud + 3 leaves — medium pluck',
    'banjhi' : 'Dormant shoot — skip, do not pluck',
    'other'  : 'Unclassified shoot',
}
# Which classes are pluckable (harvestable)
PLUCKABLE_CLASSES = {'1B-1L', '1B-1L-F', '1B-2L', '1B-2L-F', '1B-3L'}

# Colour assigned to each class for bounding boxes
CLASS_COLORS_BGR = {
    '1B-1L'  : (149, 158, 29),   # teal
    '1B-1L-F': (86, 110, 15),    # dark teal
    '1B-2L'  : (183, 74, 83),    # purple
    '1B-2L-F': (137, 52, 60),    # dark purple
    '1B-3L'  : (221, 138, 55),   # blue
    'banjhi' : (23, 117, 186),   # amber
    'other'  : (128, 135, 136),  # gray
}

# Load model once when the app starts — not on every request
MODEL_PATH = os.getenv("MODEL_PATH", "model/best.pt")
CONF       = float(os.getenv("CONF_THRESHOLD", 0.5))
IOU        = float(os.getenv("IOU_THRESHOLD", 0.45))

print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("Model loaded.")

def run_prediction(img: np.ndarray) -> dict:
    """
    Run YOLOv8 on an image and return all 3 outputs:
    1. Annotated image (base64)
    2. Detection list
    3. Summary
    """
    results = model(img, conf=CONF, iou=IOU, verbose=False)[0]

    # ── Build detection list ────────────────────────────────────────
    detections = []
    for box in results.boxes:
        cls_id   = int(box.cls)
        cls_name = CLASS_NAMES[cls_id]
        conf_val = round(float(box.conf), 3)
        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0]]

        detections.append({
            "class"           : cls_name,
            "description"     : CLASS_DESCRIPTIONS[cls_name],
            "confidence"      : conf_val,
            "confidence_level": get_confidence_color(conf_val),
            "pluckable"       : cls_name in PLUCKABLE_CLASSES,
            "bbox"            : {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    # Sort highest confidence first
    detections.sort(key=lambda x: -x["confidence"])

    # ── Build annotated image ───────────────────────────────────────
    annotated = img.copy()
    for det in detections:
        b     = det["bbox"]
        color = CLASS_COLORS_BGR[det["class"]]
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])

        # Draw box
        import cv2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label background + text
        label = f"{det['class']} {det['confidence']*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    annotated_b64 = image_to_base64(annotated)

    # ── Build summary ───────────────────────────────────────────────
    pluckable_list = [d for d in detections if d["pluckable"]]
    skipped_list   = [d for d in detections if not d["pluckable"]]

    # Find dominant flush class among pluckable detections
    top_class = pluckable_list[0]["class"] if pluckable_list else None

    summary = {
        "total_shoots"    : len(detections),
        "pluckable_count" : len(pluckable_list),
        "skip_count"      : len(skipped_list),
        "top_flush_class" : top_class,
        "top_description" : CLASS_DESCRIPTIONS.get(top_class, "—") if top_class else "—",
        "recommendation"  : _get_recommendation(pluckable_list, skipped_list),
    }

    return {
        "annotated_image": annotated_b64,
        "detections"     : detections,
        "summary"        : summary,
    }

def _get_recommendation(pluckable: list, skipped: list) -> str:
    """Generate a plain-language recommendation for the farmer."""
    if not pluckable and not skipped:
        return "No shoots detected. Try a clearer photo or lower the confidence threshold."
    if not pluckable:
        return "No pluckable shoots found. All detected shoots are dormant (banjhi)."
    if len(pluckable) >= 3:
        return f"Good yield area. {len(pluckable)} pluckable shoots ready for harvest."
    return f"{len(pluckable)} pluckable shoot(s) ready. {len(skipped)} dormant shoot(s) to skip."