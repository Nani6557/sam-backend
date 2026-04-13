from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import numpy as np
import cv2
import os
import urllib.request
from segment_anything import sam_model_registry, SamPredictor

app = FastAPI()

MODEL_PATH = "sam_vit_b_01ec64.pth"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        MODEL_PATH
    )

sam = sam_model_registry["vit_b"](
    checkpoint=MODEL_PATH
)

predictor = SamPredictor(sam)

cached_image_hash = None


@app.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...)
):
    image = Image.open(file.file).convert("RGB")
    image = image.resize((512, 512))
    image_np = np.array(image)

    h, w = image_np.shape[:2]

    mask = np.zeros((h + 2, w + 2), np.uint8)

    seed_point = (x, y)

    cv2.floodFill(
        image_np,
        mask,
        seedPoint=seed_point,
        newVal=(255, 255, 255),
        loDiff=(20, 20, 20),
        upDiff=(20, 20, 20),
    )

    filled_mask = mask[1:-1, 1:-1]

    contours, _ = cv2.findContours(
        filled_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    points = contours[0].squeeze().tolist()

    return {
        "points": points,
        "image_width": w,
        "image_height": h
    }
