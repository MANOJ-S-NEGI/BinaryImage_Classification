import os
import uuid
from fastapi import FastAPI, File, UploadFile
import uvicorn
import random
from model_dir.model_components.prediction_module import load_prep_predict_image

from fastapi.responses import FileResponse
import tensorflow as tf
import matplotlib.pyplot as plt

app = FastAPI()
img_directory = "test_images"


# main function for getting file and prediction:
def select_pred_and_image():
    image_files = os.listdir(img_directory)
    f = f"{img_directory}/{image_files[0]}"
    prediction = load_prep_predict_image(filename=f)
    return f, prediction


@app.get('/')
def read():
    return {"prediction_model_name": "Cat_or_Dog_image_classification"}


## upload the image:
@app.post('/upload')
async def create_upload_file(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    os.makedirs(img_directory, exist_ok=True)
    # save the file:
    # dir is empty only then write the file
    if len(os.listdir(img_directory)) == 0:
        with open(f"{img_directory}/{file.filename}", "wb") as f:
            f.write(contents)
    else:
        # else firest remove every thing from dir and then write file
        for f in os.listdir(img_directory):
            os.remove(f"{img_directory}/{f}")

        with open(f"{img_directory}/{file.filename}", "wb") as f:
            f.write(contents)
    return {"filename": file.filename}


##shows the recent image user uploads:
@app.post("/showimage")
async def read_image():
    f, _ = select_pred_and_image()
    return FileResponse(f)


## shows the prediction
@app.post("/imageprediction")
async def image_prediction():
    _, prediction = select_pred_and_image()
    return prediction


# shows the uploaded image with prediction as title as plot plt
@app.post("/showsimagewithprediction")
async def image_read_pred():
    f, prediction = select_pred_and_image()
    img = plt.imread(f)
    img = tf.image.resize(img, size=[150, 150])
    img = img / 255
    plt.imshow(img)  # Plot the image
    plt.title(f"Prediction: {prediction}")  # set the color to green or red
    plt.axis(False)
    plt.show()


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
