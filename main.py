import cv2
import os
import datetime

person1Video = 'Alfredo.mp4'
person2Video = 'Alfredo2.mp4'
person3Video = 'Sara2.mp4'
person4Video = 'William2.mp4'
person5Video = 'Gisell.mp4'

dataPath = os.path.dirname(os.path.realpath(__file__)) + '/data'
#dataPath = 'C:/Users/alfre/Codex/upn-reconocimiento-facial/data'
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

face_recognizer = cv2.face.EigenFaceRecognizer_create()

# Leyendo el modelo
face_recognizer.read('modeloEigenFace.xml')

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(person1Video)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


marcaciones = {'Gisell': [], 'Alfredo': [], 'Milagros': []}
while True:
    ret, frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        # EigenFaces
        if result[1] < 3600:
            if not marcaciones[imagePaths[result[0]]]:
                nombre = imagePaths[result[0]]
                hora_marcacion = datetime.datetime.now()
                print('Marcación de {}, registrada a las : {}'.format(nombre, hora_marcacion))
                marcaciones[nombre].append(hora_marcacion)
            cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)

    frame75 = rescale_frame(frame, percent=75)
    cv2.imshow("Reconocimiento Facial", frame75)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()