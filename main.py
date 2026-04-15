from fastapi import FastAPI, Body
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import base64

app = FastAPI()

cached_image = None


@app.get("/")
def home():
    return {"message": "backend working"}


@app.post("/upload-image")
async def upload_image(data: dict = Body(...)):
    global cached_image

    try:
        image_base64 = data["image"]

        image_bytes = base64.b64decode(image_base64)

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        cached_image = np.array(image)

        h, w = cached_image.shape[:2]

        return {
            "message": "uploaded successfully",
            "image_width": w,
            "image_height": h
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/segment")
async def segment(data: dict = Body(...)):
    global cached_image

    try:
        if cached_image is None:
            return {"points": []}

        x = data["x"]
        y = data["y"]

        image_np = cached_image.copy()

        h, w = image_np.shape[:2]

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        mask = np.zeros((h + 2, w + 2), np.uint8)

        image_np = cv2.GaussianBlur(
            image_np,
            (3, 3),
            0
        )

        cv2.floodFill(
            image_np,
            mask,
            seedPoint=(x, y),
            newVal=(255, 255, 255),
            loDiff=(1, 1, 1),
            upDiff=(1, 1, 1),
        )

        filled_mask = mask[1:-1, 1:-1]

        contours, _ = cv2.findContours(
            filled_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {"points": []}

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        epsilon = 0.01 * cv2.arcLength(
            largest_contour,
            True
        )

        largest_contour = cv2.approxPolyDP(
            largest_contour,
            epsilon,
            True
        )

        points = largest_contour.squeeze().tolist()

        if isinstance(points[0], int):
            points = [points]

        return {
            "points": points,
            "image_width": w,
            "image_height": h
        }

    except Exception as e:
        return {
            "error": str(e),
            "points": []
        }
