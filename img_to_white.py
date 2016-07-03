from PIL import Image
import numpy as np
import os

white=[104,125,165]
#n=90  #le righe bianche che aggiunge (2 sopra, 2 sotto, 2 sx, 2 dx)

numbers = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,32,64,90,128,256,512,1024,2048]
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
          image.save("BerrySamples_Blue/" + directory + "/" + directory + "_B_" + str(n) + "_"+ str(numb+1) + ".jpg" )
