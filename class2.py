import cv2
import matplotlib.pyplot as plt
import os 

if not os.path.isdir('artifacts'):
	os.mkdir('artifacts')
	
imagem = cv2.imread("px-people.jpg")
imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
plt.imshow(imagem)
plt.savefig('./artifacts/imagem_1.png')


imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
classificador = cv2.CascadeClassifier("classfier/haarcascade_frontalface_default.xml")
#print(help(classificador.detectMultiScale))
faces = classificador.detectMultiScale(imagem_gray, scaleFactor=1.3, minNeighbors=3)

#Para resolver a questão de escala nas imagens, existe um parâmetro do método de detecção (detectMultiScale) deste tipo de classificador, conhecido por scaleImage. Ele é capaz de redimensionar uma imagem maior até a dimensão definido no modelo. Por exemplo, se o valor deste parâmetro for 1.3, significa que a imagem a ser analisada será redimensionada (diminuir o tamanho) em 30% por etapa até alcançar a dimensão limite do modelo.

#Para maximizarmos a identificação dos rostos dos monges da imagem acima e utilizar em uma aplicação de tempo real, qual seria o melhor valor para o parâmetro scaleImage deste classificador, considerando o modelo de identificação de rosto frontal?





len(faces)
#5
#faces[0]
#array([1088,  102,  101,  101], dtype=int32)
imagem_anotada = imagem.copy()
for (x,y,w,h) in faces:
    cv2.rectangle(imagem_anotada, (x,y), (x+w, y+h), (255, 255, 0), 2)
plt.figure(figsize=(20,10))
plt.imshow(imagem_anotada)
plt.savefig('./artifacts/face_detected.png')
#<matplotlib.image.AxesImage at 0x12233d128>

face_imagem = 0

for (x,y,w,h) in faces:
    face_imagem += 1
    imagem_roi = imagem[y:y+h, x:x+w]
    imagem_roi = cv2.cvtColor(imagem_roi, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./artifacts/face_" + str(face_imagem) + ".png", imagem_roi)
