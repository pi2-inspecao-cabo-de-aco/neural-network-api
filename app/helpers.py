import warnings
from PIL import Image
import io

def ignore_wanings():
  warnings.filterwarnings('ignore', category=Warning)

def read_byte_img(image_path):
  img = Image.open(image_path, mode='r')
  img_by_arr = io.BytesIO()
  img.save(img_by_arr, format='PNG')

  return img_by_arr