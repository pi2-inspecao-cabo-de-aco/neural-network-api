from fastapi import FastAPI
from fastai import *
from fastai.vision import *
from starlette.responses import JSONResponse
from os.path import dirname, abspath, join
from helpers import ignore_wanings

ignore_wanings()

model_path = Path('./models/resnet50')
root_path = dirname(abspath(__file__))
images_path = join(root_path, 'public/images/')
learn = load_learner(model_path, 'model.pkl')

print(root_path)
print(images_path)

app = FastAPI(
  title = 'Neural Network API',
  description = 'Projeto referente à API da Rede Neural do projeto Inspeção de Cabo de Aço',
  version = '1.0'
)

@app.get('/')
def root():
  return {'Inspeção de Cabo de Aço - Neural Network API'}

@app.post('/analyze/{image_name}')
def analyze(image_name: str):
  """Returns the classification of the condition of cable using the neural network, given an image"""
  img = images_path + image_name + '.png'
  img_to_predict = open_image(img)
  prediction = learn.predict(img_to_predict)[0]

  return JSONResponse({'condition': str(prediction)})
