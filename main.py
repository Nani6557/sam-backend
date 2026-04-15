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


@app.post("/segment")
async def segment(data: dict = Body(...)):
    global cached_image

    try:
        if cached_image is None:
            return {"points": []}

        image_np = cached_image.copy()

        h, w = image_np.shape[:2]

        gray = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2GRAY
        )

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            blurred,
            50,
            150
        )

        contours, _ = cv2.findContours(
            edges,
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

        approx = cv2.approxPolyDP(
            largest_contour,
            epsilon,
            True
        )

        points = approx.squeeze().tolist()

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
