# Reconocimiento Facial Python

Sistema de marcación biométrica

## Dependencias
* Python 3
* opencv-python
* imutils
* opencv-contrib-python

## Instalando dependencias
```bash
pip3 install -r requirements.txt
```

## Uso

Si se usa el método de captura de imágenes y reconocimiento via camara web omitir el paso 1

1. Agregar videos de las personas a reconocer en el directorio _videos_

2. Cambiar el nombre de la persona a reconocer en el archivos de captura 
de rostros
 
3. Ejecutar los siguientes scripts secuencialmente

### Captura de imágenes de rostros para alimentar el modelo

src/capturar_rostros.py

### Generación modelo de reconocimiento facial

src/generar_modelo.py

### Reconocimiento facial

main.py