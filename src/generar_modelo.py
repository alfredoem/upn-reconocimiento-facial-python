import cv2
import os
import numpy as np


class GenerarModelo:
    DATASET_ROSTROS = os.path.dirname(os.path.realpath(__file__)) + '/../data'
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
            self.contador_etiquetas = self.contador_etiquetas + 1

    def generar(self):
        self.preparar_dataset()
        face_recognizer = cv2.face.EigenFaceRecognizer_create()
        print("Generando modelo...")
        face_recognizer.train(self.lista_rostros, np.array(self.lista_etiquetas))
        face_recognizer.write('modelo_reco_entrenado.xml')
        print("Modelo generado!")


generador = GenerarModelo()
generador.generar()