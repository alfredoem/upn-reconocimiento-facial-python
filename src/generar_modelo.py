import cv2
import os
import numpy as np


class GenerarModelo:
    DATASET_ROSTROS = os.path.dirname(os.path.realpath(__file__)) + os.sep + '..' + os.sep + 'data'
    LISTA_PERSONAS = os.listdir(DATASET_ROSTROS)

    def __init__(self):
        self.lista_etiquetas = []
        self.contador_etiquetas = 0
        self.lista_rostros = []

    def preparar_dataset(self):
        # Recorremos las capturas de rostros de cada persona
        for nombre in self.LISTA_PERSONAS:
            dataset_persona = self.DATASET_ROSTROS + os.sep + nombre
            for imagen in os.listdir(dataset_persona):
                # Agregamos una etiqueta númerica por cada persona a a lista de etiquetas
                self.lista_etiquetas.append(self.contador_etiquetas)
                # Agregamos la imagen en escala de grises a la lista de rostros
                self.lista_rostros.append(cv2.imread(dataset_persona + os.sep + imagen, 0))
            self.contador_etiquetas = self.contador_etiquetas + 1

    def generar(self):
        # Preparamos el dataset para genera el modelo de reconocimiento de las personas
        self.preparar_dataset()
        print("Generando modelo...")
        # Generamos el modelo enviando la lista de rostros y la lista de etiquetas
        face_recognizer = cv2.face.EigenFaceRecognizer_create()
        face_recognizer.train(self.lista_rostros, np.array(self.lista_etiquetas))
        # Almacenamos el modelo entrenado con las capturas de rostros de las personas
        face_recognizer.write('modelo_reco_entrenado.xml')
        print("Modelo generado!")


generador = GenerarModelo()
generador.generar()