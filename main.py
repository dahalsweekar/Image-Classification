import io

from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from PIL import Image
import numpy as np

app = FastAPI()
loaded_model = tf.keras.models.load_model('./model/my_model.h5')


@app.post('/predict/')
async def predict_image(image: UploadFile = File(...)):
    contents = await image.read()
    pil_img = Image.open(io.BytesIO(contents)).convert('L')
    img = pil_img.resize((224, 224))
    img_rgb = Image.new('RGB', img.size)
    img_rgb.paste(img)
    img_array = np.array(img_rgb) / 255.0

    input_image = np.expand_dims(img_array, axis=0)

    predictions = loaded_model.predict(input_image)
    return {"predictions": predictions[0].item()}
