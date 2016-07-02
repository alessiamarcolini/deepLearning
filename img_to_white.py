from PIL import Image
import numpy as np
import os

white=[255,255,255]
#n=90  #le righe bianche che aggiunge (2 sopra, 2 sotto, 2 sx, 2 dx)

numbers = range(1,16)
directories = ["early", "good", "late"]


for n in numbers:
    for directory in directories:
        path = 'BerrySamples_Original/'
        paths = os.listdir(path + "/" + directory)
        paths.sort()

        for numb, name in enumerate(paths):



          img = Image.open(path + directory + "/" + name )
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
          image.save("BerrySamples_White/" + directory + "/" + directory + "_W_" + str(n) + "_"+ str(numb+1) + ".jpg" )
