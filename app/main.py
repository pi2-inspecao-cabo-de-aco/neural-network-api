from fastapi import FastAPI
from fastai import *
from fastai.vision import *
from starlette.responses import JSONResponse
from os.path import dirname, abspath, join
from helpers import ignore_wanings, read_byte_img

ignore_wanings()

model_path = Path('./models/resnet50')
dirname = dirname(dirname(abspath(__file__)))
images_path = join(dirname, 'public/')
learn = load_learner(model_path, 'model.pkl')

app = FastAPI()

@app.get('/')
def root():
  return {'Inspeção de Cabo de Aço - Neural Network API'}

@app.post('/analyze')
def analyze(image_name: str):
  image_name = image_name + '.png'
  img = images_path + image_name
  byte_img = read_byte_img(img)
  prediction = learn.predict(byte_img)[0]

  return JSONResponse({'result': str(prediction)})