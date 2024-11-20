import cv2
import matplotlib.pyplot as plt
import os 

if not os.path.isdir('artifacts'):
	os.mkdir('artifacts')
	
imagem = cv2.imread("px-girl.jpg")
plt.imshow(imagem)
#<matplotlib.image.AxesImage at 0x115f61c18>

imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
plt.imshow(imagem_rgb)
plt.savefig('./artifacts/imagem_rgb.png')
#<matplotlib.image.AxesImage at 0x1121c5860>

imagem_rgb.shape
#(853, 1280, 3)
imagem_gray = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2GRAY)
plt.imshow(imagem_gray, cmap="gray")
plt.savefig('./artifacts/imagem_cinza.png')
#<matplotlib.image.AxesImage at 0x112214438>

#imagem_gray.shape


imagem = cv2.imread("px-people.jpg")
imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
#plt.imshow(imagem)
#<matplotlib.image.AxesImage at 0x1130031d0>

#imagem.shape
#(853, 1280, 3)
imagem_roi = imagem[100:200, 1000:1200]
plt.imshow(imagem_roi)
plt.savefig('./artifacts/imagem_roi.png')
#<matplotlib.image.AxesImage at 0x11652db70>

plt.savefig("./artifacts/imagem_roi.png", imagem_roi)
#True
imagem_roi_bgr = cv2.cvtColor(imagem_roi, cv2.COLOR_RGB2BGR)
cv2.imwrite("imagem_roi2.png", imagem_roi_bgr)
#plt.savefig('imagem_roi.png')

