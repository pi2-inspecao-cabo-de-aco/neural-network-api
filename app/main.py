from fastapi import FastAPI
from fastai import *
from fastai.vision import *
from starlette.responses import JSONResponse
from os.path import dirname, abspath, join
from helper import ignore_wanings
from starlette.middleware.cors import CORSMiddleware
import re
from pydantic import BaseModel

ignore_wanings()

model_path = Path('./models/resnet50')
root_path = dirname(abspath(__file__))
images_path = join(root_path, 'public/images/')
learn = load_learner(model_path, 'model.pkl')

app = FastAPI(
  title = 'Neural Network API',
  description = 'Projeto referente à API da Rede Neural do projeto Inspeção de Cabo de Aço',
  version = '1.0'
)

origins = [
  'http://localhost:8080',
  'http://localhost:4000',
  'http://localhost:3030'
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)

class Body(BaseModel):
  img: str

@app.get('/')
def root():
  return {'Inspeção de Cabo de Aço - Neural Network API'}

@app.post('/analyze')
async def analyze(*, body: Body):
  """Returns the classification of the condition of cable using the neural network, given an image"""
  img = body.img
  image = root_path + img
  img_to_predict = open_image(image)
  prediction = learn.predict(img_to_predict)[0]

  return JSONResponse({'condition': str(prediction)})
