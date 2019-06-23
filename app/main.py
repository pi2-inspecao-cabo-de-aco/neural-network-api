from fastapi import FastAPI
from fastai import *
from fastai.vision import *
from starlette.responses import JSONResponse
import io
from os.path import dirname, abspath, join
from PIL import Image

model_path = Path('./models/resnet50')

dirname = dirname(dirname(abspath(__file__)))
images_path = join(dirname, '/public/')

app = FastAPI()

learn = load_learner(model_path, 'model.pkl')

def read_byte_img(image_path):
  img = Image.open(image_path, mode='r')
  imgByteArr = io.BytesIO()
  img.save(imgByteArr, format='PNG')

  return imgByteArr

@app.get("/")
def root():
  return {"Inspeção de Cabo de Aço - Neural Network API"}

@app.post("/analyze")
def analyze(image_name: str):
  image_name = image_name + '.png'
  img = images_path + image_name
  byte_img = read_byte_img(img)
  prediction = learn.predict(byte_img)[0]

  return JSONResponse({'result': str(prediction)})