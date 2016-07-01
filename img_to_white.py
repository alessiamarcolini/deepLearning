from PIL import Image
import numpy as np
import os

white=[255,255,255]
n=2048 #le righe bianche che aggiunge (2 sopra, 2 sotto, 2 sx, 2 dx)

path = 'BerrySamples_Original/early'
paths = os.listdir(path)
paths.sort()

for numb, name in enumerate(paths):


  img = Image.open(path + name)
  arr = np.array(img)
  size=arr.shape

  whiterow=[white]*(size[1]+n*2)

  matrix=[whiterow]*n


  for i in range(size[0]):
  	line=(([white]*n)+(arr[i].tolist())+([white]*n))
  	matrix.append(line)

  for i in range(n):
  	matrix.append(whiterow)


  newarr=np.array(matrix, dtype=np.uint8)

  image = Image.fromarray(newarr,'RGB')
  image.save("early_W_" + str(numb+1) + ".jpg" )
