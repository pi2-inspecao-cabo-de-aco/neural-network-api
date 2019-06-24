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

print(images_path)

app = FastAPI(
  title = 'Neural Network API',
  description = 'Projeto referente à API da Rede Neural do projeto Inspeção de Cabo de Aço',
  version = '1.0'
)

@app.get('/')
def root():
  return {'Inspeção de Cabo de Aço - Neural Network API'}

@app.post('/analyze', )
async def analyze(image_name: str):
  """Returns the classification of the state of the cable using the neural network, given an image"""
  image_name = image_name + '.png'
  img = images_path + image_name
  byte_img = read_byte_img(img)
  prediction = learn.predict(byte_img)[0]

  return JSONResponse({'result': str(prediction)})