from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

app = FastAPI()

sam = sam_model_registry["vit_b"](
    checkpoint="sam_vit_b_01ec64.pth"
)

predictor = SamPredictor(sam)

cached_image_hash = None


@app.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...)
):
    global cached_image_hash

    image = Image.open(file.file).convert("RGB")
    image = image.resize((512, 512))
    image_np = np.array(image)

    current_hash = hash(image_np.tobytes())

    if cached_image_hash != current_hash:
        predictor.set_image(image_np)
        cached_image_hash = current_hash

    input_point = np.array([[x, y]])
    input_label = np.array([1])

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    mask = masks[0].astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    points = contours[0].squeeze().tolist()

    h, w = image_np.shape[:2]

    return {
        "points": points,
        "image_width": w,
        "image_height": h
    }