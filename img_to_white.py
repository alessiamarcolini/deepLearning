from PIL import Image
import numpy as np
white=[255,255,255]
n=2048 #le righe bianche che aggiunge (2 sopra, 2 sotto, 2 sx, 2 dx)

img = Image.open("god.jpg")
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
image.save("output.jpg")
