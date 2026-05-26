
from fastapi import Body
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import base64

@app.post("/segment")
async def segment(data: dict = Body(...)):
    try:
        image_base64 = data["image"]

        x = data["x"]
        y = data["y"]

        image_bytes = base64.b64decode(
            image_base64
        )

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        image_np = np.array(image)

        h, w = image_np.shape[:2]

        x = int(x)
        y = int(y)

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        print("IMAGE SIZE:", w, h)
        print("CLICK:", x, y)

        flood_image = image_np.copy()

        mask = np.zeros(
            (h + 2, w + 2),
            np.uint8
        )

        cv2.floodFill(
            flood_image,
            mask,
            seedPoint=(x, y),
            newVal=(255, 255, 255),
            loDiff=(35, 35, 35),
            upDiff=(35, 35, 35),
        )

        filled_mask = mask[1:-1, 1:-1]

        print(
            "MASK SUM:",
            filled_mask.sum()
        )

        contours, _ = cv2.findContours(
            filled_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        print(
            "CONTOURS:",
            len(contours)
        )

        if len(contours) == 0:
            return {
                "points": []
            }

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        epsilon = (
            0.002
            * cv2.arcLength(
                largest_contour,
                True
            )
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
        print("ERROR:", str(e))

        return {
            "error": str(e),
            "points": []
        }

