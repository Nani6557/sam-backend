from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

app = FastAPI()


@app.get("/")
def home():
    return {"message": "backend working"}


@app.post("/segment")
async def segment(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...)
):
    try:
        # Read uploaded image bytes
        contents = await file.read()

        print("REQUEST RECEIVED")
        print("filename:", file.filename)
        print("x:", x)
        print("y:", y)
        print("file size:", len(contents))

        # Open image safely from bytes
        image = Image.open(BytesIO(contents)).convert("RGB")

        # Resize image for processing
        image = image.resize((512, 512))

        # Convert to numpy array
        image_np = np.array(image)

        h, w = image_np.shape[:2]

        # Create mask for flood fill
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Keep touch point inside bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        seed_point = (x, y)

        # Flood fill region
        cv2.floodFill(
            image_np,
            mask,
            seedPoint=seed_point,
            newVal=(255, 255, 255),
            loDiff=(10, 10, 10),
            upDiff=(10, 10, 10),
        )

        # Remove border padding from mask
        filled_mask = mask[1:-1, 1:-1]

        # Find contours
        contours, _ = cv2.findContours(
            filled_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # If nothing detected
        if len(contours) == 0:
            return {"points": []}

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        points = largest_contour.squeeze().tolist()

        # Ensure points always return as list
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
