from src.CycleGanModel import *
from src.utils import *

# dataset path
path = ''
# load dataset A
dataA = load_images_and_normalize(path + 'dataset_monet/')
print('Loaded dataA: ', dataA.shape)
# load dataset B
dataB = load_images_and_normalize(path + 'dataset_content/')
print('Loaded dataB: ', dataB.shape)

cyclegan = CycleGan(dataA[0].shape)
cyclegan.train(dataA, dataB, epochs=100)

