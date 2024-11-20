import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os import listdir, path, makedirs
from os.path import isfile, join
import shutil

def padronizar_imagem(imagem_caminho):
    """
    Redimensionar as imagens para um valor próximo da mediana do tamanho de todas as imagens, de tal forma que não seja nem tão grande nem tão pequeno.
    """
    imagem = cv2.imread(imagem_caminho, cv2.IMREAD_GRAYSCALE)
    imagem = cv2.resize(imagem, (200, 200), interpolation=cv2.INTER_LANCZOS4)
    return imagem


if not os.path.isdir('artifacts'):
    os.mkdir('artifacts')

imagem_face_1 = cv2.imread("./cropped_faces/s01_01.jpg")
imagem_face_1 = cv2.cvtColor(imagem_face_1, cv2.COLOR_BGR2RGB)

imagem_face_2 = cv2.imread("./cropped_faces/s02_01.jpg")
imagem_face_2 = cv2.cvtColor(imagem_face_2, cv2.COLOR_BGR2RGB)

imagem_face_3 = cv2.imread("./cropped_faces/s03_01.jpg")
imagem_face_3 = cv2.cvtColor(imagem_face_3, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,10))
plt.subplot(131)
plt.title("Sujeito 01")
plt.imshow(imagem_face_1)
plt.subplot(132)
plt.title("Sujeito 02")
plt.imshow(imagem_face_2)
plt.subplot(133)
plt.title("Sujeito 03")
plt.imshow(imagem_face_3)

#plt.show()
plt.savefig("./artifacts/sujeitos.png")

#imagem_face_1.shape
#(241, 181, 3)
#imagem_face_2.shape
#(211, 141, 3)
#imagem_face_3.shape

faces_caminho = "./cropped_faces/"
lista_arq_faces = [f for f in listdir(faces_caminho) if isfile(join(faces_caminho, f))]

faces_path_treino = "imagens/treino/"
faces_path_teste = "imagens/teste/"

if not path.exists(faces_path_treino):
    makedirs(faces_path_treino)

if not path.exists(faces_path_teste):
    makedirs(faces_path_teste)

for arq in lista_arq_faces:
    sujeito = arq[1:3]
    numero = arq[4:6]
    
    if int(numero) <= 10:
        shutil.copyfile(faces_caminho + arq, faces_path_treino + arq)
    else:
        shutil.copyfile(faces_caminho + arq, faces_path_teste + arq)
        
lista_faces_treino = [f for f in listdir(faces_path_treino) if isfile(join(faces_path_treino, f))]
lista_faces_teste = [f for f in listdir(faces_path_teste) if isfile(join(faces_path_teste, f))]
dados_treinamento, sujeitos = [], []

for i, arq in enumerate(lista_faces_treino):
    imagem_path = faces_path_treino + arq
    imagem = padronizar_imagem(imagem_path)
    dados_treinamento.append(imagem)
    sujeito = arq[1:3]
    sujeitos.append(int(sujeito))

dados_teste, sujeitos_teste = [], [] 


for i, arq in enumerate(lista_faces_teste):
    imagem_path = faces_path_teste + arq
    imagem = padronizar_imagem(imagem_path)
    dados_teste.append(imagem)
    sujeito = arq[1:3]
    sujeitos_teste.append(int(sujeito))

plt.figure(figsize=(20,10))
plt.imshow(dados_treinamento[0], cmap="gray")
plt.title(sujeito[0])
plt.savefig("./artifacts/sujeitos1.png")

plt.figure(figsize=(20,10))
plt.imshow(dados_teste[0], cmap="gray")
plt.title(sujeitos_teste[0])
plt.savefig("./artifacts/sujeitos2.png")

sujeitos = np.asarray(sujeitos, dtype=np.int32)
sujeitos_teste = np.asarray(sujeitos_teste, dtype=np.int32)
modelo_eingenfaces = cv2.face.EigenFaceRecognizer_create()
modelo_eingenfaces.train(dados_treinamento, sujeitos)
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.title("Sujeito " + str(sujeitos_teste[6]))
plt.imshow(dados_teste[6], cmap="gray")
plt.subplot(122)
plt.title("Sujeito " + str(sujeitos_teste[7]))
plt.imshow(dados_teste[7], cmap="gray")
plt.savefig(f"./artifacts/eingenfaces.png")

print("Eigenfaces")
predicao = modelo_eingenfaces.predict(dados_teste[6])
print(predicao)
predicao = modelo_eingenfaces.predict(dados_teste[7])
print(predicao)


modelo_fisherfaces = cv2.face.FisherFaceRecognizer_create()
modelo_fisherfaces.train(dados_treinamento, sujeitos)
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.title("Sujeito " + str(sujeitos_teste[13]))
plt.imshow(dados_teste[13], cmap="gray")

plt.subplot(122)
plt.title("Sujeito " + str(sujeitos_teste[19]))
plt.imshow(dados_teste[19], cmap="gray")
plt.savefig(f"./artifacts/fisherfaces.png")

print("Fisherfaces")
predicao = modelo_fisherfaces.predict(dados_teste[13])
print(predicao)
predicao = modelo_fisherfaces.predict(dados_teste[19])
print(predicao)


modelo_lbph = cv2.face.LBPHFaceRecognizer_create()
modelo_lbph.train(dados_treinamento, sujeitos)
plt.figure(figsize=(20,10))

plt.subplot(121)
plt.title("Sujeito " + str(sujeitos_teste[21]))
plt.imshow(dados_teste[21], cmap="gray")

plt.subplot(122)
plt.title("Sujeito " + str(sujeitos_teste[27]))
plt.imshow(dados_teste[27], cmap="gray")
plt.savefig(f"./artifacts/LBPH.png")

print("LBPH")
predicao = modelo_lbph.predict(dados_teste[21])
print(predicao)
predicao = modelo_lbph.predict(dados_teste[27])
print(predicao)


y_pred_eingenfaces = []
for item in dados_teste:
    y_pred_eingenfaces.append(modelo_eingenfaces.predict(item)[0])
acuracia_eingenfaces = accuracy_score(sujeitos_teste, y_pred_eingenfaces)
print(f"Eigenfaces accuracy: {acuracia_eingenfaces}")

y_pred_fisherfaces = []
for item in dados_teste:
    y_pred_fisherfaces.append(modelo_fisherfaces.predict(item)[0])
acuracia_fisherfaces = accuracy_score(sujeitos_teste, y_pred_fisherfaces)

y_pred_lbph = []
for item in dados_teste:
    y_pred_lbph.append(modelo_lbph.predict(item)[0])
acuracia_lbph = accuracy_score(sujeitos_teste, y_pred_lbph)
print(f"Eigenfaces accuracy: {acuracia_lbph}")
