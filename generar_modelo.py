import cv2
import os
import numpy as np


class GenerarModelo:
    DATASET_ROSTROS = dataPath = os.path.dirname(os.path.realpath(__file__)) + '/data'
    LISTA_PERSONAS = os.listdir(DATASET_ROSTROS)

    def __init__(self):
        self.lista_etiquetas = []
        self.contador_etiquetas = 0
        self.lista_rostros = []

    def preparar_dataset(self):
        for nombre in self.LISTA_PERSONAS:
            dataset_persona = self.DATASET_ROSTROS + '/' + nombre
            for imagen in os.listdir(dataset_persona):
                self.lista_etiquetas.append(self.contador_etiquetas)
                self.lista_rostros.append(cv2.imread(dataset_persona + '/' + imagen, 0))
                image = cv2.imread(dataset_persona + '/' + imagen, 0)
            self.contador_etiquetas = self.contador_etiquetas + 1

    def generar(self):
        self.preparar_dataset()
        face_recognizer = cv2.face.EigenFaceRecognizer_create()
        print("Generando modelo...")
        face_recognizer.train(self.lista_rostros, np.array(self.lista_etiquetas))
        face_recognizer.write('modeloEigenFace.xml')
        print("Modelo generado!")


generador = GenerarModelo()
generador.generar()

#dataPath = 'C:/Users/alfre/Codex/upn-reconocimiento-facial/data'
# dataPath = os.path.dirname(os.path.realpath(__file__)) + '/data'
# peopleList = os.listdir(dataPath)
# print('Lista de personas: ', peopleList)
#
# labels = []
# facesData = []
# label = 0
#
# for nameDir in peopleList:
#     personPath = dataPath + '/' + nameDir
#     print('Leyendo las imágenes')
#
#     for fileName in os.listdir(personPath):
#         print('Rostros: ', nameDir + '/' + fileName)
#         labels.append(label)
#         facesData.append(cv2.imread(personPath + '/' + fileName, 0))
#         image = cv2.imread(personPath+'/'+fileName,0)
#     label = label + 1
#
# # Métodos para entrenar el reconocedor
# face_recognizer = cv2.face.EigenFaceRecognizer_create()
#
# # Entrenando el reconocedor de rostros
# print("Entrenando...")
# face_recognizer.train(facesData, np.array(labels))
#
# # Almacenando el modelo obtenido
# face_recognizer.write('modeloEigenFace.xml')
#
# print("Modelo almacenado...")
