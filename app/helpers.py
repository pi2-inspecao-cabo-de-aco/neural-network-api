import warnings

def ignore_wanings():
  warnings.filterwarnings('ignore', category=Warning)