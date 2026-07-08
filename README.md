# COLOR RECOGNITION

This project focuses on color classifying by K-Nearest Neighbors Machine Learning Classifier which is trained by R, G, B Color Histogram. It can classify White, Black, Red, Green, Blue, Orange, Yellow and Violet. If you want to classify more color or improve the accuracy you should work on the [training data](https://github.com/ahmetozlu/color_classifier/tree/master/src/training_dataset) or consider about other color features such as [Color Moments](https://en.wikipedia.org/wiki/Color_moments) or [Color Correlogram](http://www.cs.cornell.edu/rdz/Papers/ecdl2/spatial.htm).

You can use [color_recognition_api](https://github.com/ahmetozlu/color_recognition/tree/master/src/color_recognition_api) to perform real-time color recognition in your projects. You can find a sample usage of [color_recognition_api](https://github.com/ahmetozlu/color_recognition/tree/master/src/color_recognition_api) in this [**repo**](https://github.com/ahmetozlu/vehicle_counting_tensorflow). ***Please contact if you need professional color recognition project with the super high accuracy!***

## Quick Start (Inicio Rápido)

**Windows:**
1. Ejecuta `install.bat` (solo la primera vez)
2. Luego ejecuta `run_gui.bat`

**macOS/Linux:**
1. Ejecuta `bash install.sh` (solo la primera vez)
2. Luego ejecuta `bash run_gui.sh`

¡Sin necesidad de entorno virtual! Las dependencias se instalan automáticamente.

## Quick Demo

***Run [color_classification_webcam.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_classification_webcam.py) to perform real-time color recognition on a webcam stream.***

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/34917659-8497acae-f95a-11e7-93fb-f7cd6cc3128a.gif">
</p>

***Run [color_classification_image.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_classification_image.py) to perform color recognition on a single image.***

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42423806-14cdfa7a-8309-11e8-9478-23d50fc0002f.png">
</p>

---

## Instrucciones de uso

### Opcion 1: Forma mas facil (recomendada - Sin entorno virtual)

#### Primer paso: Instalar dependencias

**Windows:**
```bash
install.bat
```

**macOS/Linux:**
```bash
bash install.sh
```

Este script verifica que tengas Python instalado e instala automaticamente todas las dependencias necesarias.

#### Segundo paso: Ejecutar el programa

**Windows:**
```bash
run_gui.bat
```

**macOS/Linux:**
```bash
bash run_gui.sh
```

El programa se ejecutara sin necesidad de entorno virtual. ¡Facilisimo para compartir con otros!

> **Nota:** La primera vez que ejecutes el programa puede tardar unos segundos porque genera automaticamente el archivo `training.data` a partir de las imagenes en `src/training_dataset/`. Las siguientes ejecuciones son instantaneas.

### Opcion 2: Instalacion manual (con entorno virtual)

Si prefieres mayor control o usar entorno virtual:

```bash
pip install -r requirements.txt
cd src
python color_classification_gui.py
```

### Interfaz grafica: `color_classification_gui.py`

Es la forma mas facil de usar el proyecto. Tiene un boton **"Ayuda / Instrucciones"** en la parte superior que muestra en cualquier momento la guia completa dentro de la propia app. En resumen:

- **Pestana "Imagen":** sube una foto con "Subir imagen...", opcionalmente arrastra el mouse sobre ella para marcar solo el objeto (asi el fondo no se mezcla en la lectura) y pulsa "Detectar color". El resultado se muestra con su **porcentaje de confianza**.
- **Pestana "Camara":**
  1. Pulsa **"Iniciar camara"**.
  2. Coloca el objeto a identificar **dentro del recuadro verde central**, ocupando la mayor parte posible del recuadro.
  3. Mira la barra **"Enfoque"**: si esta en naranja/rojo, la lectura no es confiable. Acerca o aleja el objeto hasta que la camara enfoque, mejora la iluminacion, o pulsa **"Reenfocar"** para forzar una recalibracion del foco (util si la camara no enfoca sola, algo comun con DroidCam o camaras USB baratas).
  4. La prediccion aparece debajo del video junto a su confianza (%). Si la confianza es baja, la lectura se ignora automaticamente en vez de mostrar un color al azar.
  5. Desmarca "Analizar solo el centro" si prefieres que se analice el frame completo (no recomendado si hay varios objetos/colores dentro de camara).
  6. Pulsa **"Detener camara"** para apagarla.

### 3. Deteccion en vivo por linea de comandos: `color_classification_webcam.py`

```bash
cd src
python color_classification_webcam.py
```

Abre una ventana con el video de la camara. Coloca el objeto dentro del **recuadro central** que se dibuja en pantalla; ahi es donde se lee el color, no en el resto del frame. Debajo de la prediccion (que incluye su % de confianza) se muestra una barra de nitidez en vivo.

Controles de teclado (tambien listados en la parte inferior de la ventana):

| Tecla | Accion |
|---|---|
| `f` | Recalibra el enfoque manualmente |
| `q` | Cierra el programa |

### 4. Deteccion sobre una imagen suelta: `color_classification_image.py`

```bash
cd src
python color_classification_image.py ruta/a/tu/imagen.jpg
```

Si tienes entorno grafico disponible, primero se abre una ventana para que arrastres el mouse y selecciones la region exacta a analizar (ENTER confirma, ESC usa la imagen completa). El resultado se muestra en una ventana con la prediccion y su confianza; si no hay entorno grafico (por ejemplo al correrlo en un servidor por SSH), se analiza la imagen completa y el resultado se guarda como `resultado.jpg` en la carpeta `src/`.

### Consejos para mejores resultados

- **Iluminacion uniforme:** evita contraluz y sombras fuertes sobre el objeto; la luz natural difusa o luz blanca uniforme da mejores lecturas que luces de colores o muy tenues.
- **Evita superficies muy brillantes o reflectantes:** los reflejos especulares se descartan automaticamente, pero un objeto extremadamente brilloso igual puede confundir la lectura.
- **Llena el recuadro:** entre mas espacio del recuadro central ocupe el objeto (y menos fondo se cuele), mas precisa es la deteccion.
- **Dale un segundo a que se estabilice:** la prediccion se calcula promediando varias lecturas recientes para evitar que "parpadee" entre colores; mantener el objeto quieto un momento mejora el resultado.
- **Si la camara no enfoca:** usa el boton "Reenfocar" (GUI) o la tecla `f` (script de webcam) para forzar una recalibracion manual del foco.
- **Agregar o mejorar colores:** si un color se confunde seguido, agrega mas fotos representativas a la carpeta correspondiente dentro de `src/training_dataset/` (o crea una carpeta nueva para un color que no exista todavia) y borra `src/training.data` para que se regenere con los nuevos ejemplos.

---

**What does this program do?**
1. **Feature Extraction:** Perform feature extraction for getting the R, G, B Color Histogram values of [training images](https://github.com/ahmetozlu/color_classifier/tree/master/src/training_dataset)
2. **Training K-Nearest Neighbors Classifier:** Train KNN classifier by R, G, B Color Histogram values
3. **Classifying by Trained KNN:** Read Web Cam frame by frame, perform feature extraction on each frame and then classify the mean color of it by trained KNN classifier.
---

**TODOs:**

- "Add New Color" utility will be added.
- New feature extractors will be added.
- New classifiers will be added.


## Theory

In this study, colors are classified by using K-Neares Neşghbor Machine Learning classifier algorithm. This classifier is trained by image R, G, B Color Histogram values. The general work flow is given at the below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/35335133-a9632c70-0125-11e8-9204-0b4bfd0702a7.png" {width=35px height=350px}>
</p>

You should know 2 main pheomena to understand basic Object Detection/Recognition Systems of Computer Vision and Machine Learning.

**1.) Feature Extraction**

How to represent the interesting points we found to compare them with other interesting points (features) in the image.

**2.) Classification**

An algorithm that implements classification, especially in a concrete implementation, is known as a classifier. The term "classifier" sometimes also refers to the mathematical function, implemented by a classification algorithm, that maps input data to a category.

For this project;

**1.) Feature Extraction** = Color Histogram

Color Histogram is a representation of the distribution of colors in an image. For digital images, a color histogram represents the number of pixels that have colors in each of a fixed list of color ranges, that span the image's color space, the set of all possible colors.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/34918867-44f5feaa-f96b-11e7-9994-1747846266c9.png">
</p>

**2.) Classification** = K-Nearest Neighbors Algorithm

K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions). KNN has been used in statistical estimation and pattern recognition already in the beginning of 1970’s as a non-parametric technique.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/34918895-c7b94d24-f96b-11e7-87da-8619d9bd4246.png">
</p>

## Implementation

[OpenCV](https://pypi.python.org/pypi/opencv-python) was used for color histogram calculations and knn classifier. [NumPy](https://stackoverflow.com/questions/29499815/how-to-install-numpy-on-windows-using-pip-install) was used for matrix/n-dimensional array calculations. The program was developed on Python at Linux environment.

In the “[src](https://github.com/ahmetozlu/color_recognition/tree/master/src)” folder, there are 2 Python classes which are:

- **[color_classification_webcam.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_classification_webcam.py):** test class to perform real-time color recognition form webcam stream.

- **[color_classification_image.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_classification_image.py):** test class to perform color recognition on a single image.

In the “[color_recognition_api](https://github.com/ahmetozlu/color_recognition/tree/master/src/color_recognition_api)” folder, there are 2 Python classes which are:

- **[feature_extraction.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_recognition_api/color_histogram_feature_extraction.py):** feature extraction operation class

- **[knn_classifier.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_recognition_api/knn_classifier.py):** knn classification class

**1.) Explanation of “[feature_extraction.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_recognition_api/color_histogram_feature_extraction.py)"**

I can get the RGB color histogram of images by this Python class. For example, plot of RGB color histogram for one of the red images is given at the below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/34919478-f198beb8-f975-11e7-8c1c-0a552f7cd673.jpg" {width=25px height=250px}>
</p>

I decided to use bin number of histogram which has the peak value of pixel count for R, G and B as feature so I can get the dominant R, G and B values to create feature vectors for training. For example, the dominant R, G and B values of the red image which is given at above is [254, 0, 2].

I get the dominant R, G, B values by using Color Histogram for each training image then I labelled them because KNN classifier is a supervised learner and I deploy these feature vectors in the csv file. Thus, I create my training feature vector dataset. It can be found in the file which name’s is [training.data](https://github.com/ahmetozlu/color_recognition/blob/master/src/training.data) under src folder.

**2.) Explanation of “[knn_classifier.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_recognition_api/knn_classifier.py)”**

This class provides these main calculations;

1. Fetching training data
2. Fetching test image features
3. Calculating euclidean distance
4. Getting k nearest neighbors
5. Prediction of color
6. Returning the prediction is true or false

**“[color_classification_webcam.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_classification_webcam.py)”** is the main class of my program, it provides;

1. Calling [feature_extraction.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_recognition_api/color_histogram_feature_extraction.py) to create training data by feature extraction
2. Calling [knn_classifier.py](https://github.com/ahmetozlu/color_recognition/blob/master/src/color_recognition_api/knn_classifier.py) for classification

You can find training data in [here](https://github.com/ahmetozlu/color_classifier/tree/master/src/training_dataset).

You can find features are got from training data in [here](https://raw.githubusercontent.com/ahmetozlu/color_classifier/master/src/training.data).

## Conclusion

I think, the training data has a huge important in classification accuracy. I created my training data carefully but maybe the accuracy can be higher with more suitable training data.

Another important thing is lightning and shadows. In my test images, the images which were taken under bad lighting conditions and with shadows are classified wrong (false positives), maybe some filtering algorithm should/can be implemented before the test images send to KNN classifier Thus, accuracy can be improved.

### Mejoras aplicadas a esta version

Sobre la version original, este proyecto ahora incluye:

- **Extraccion de color mas robusta ante iluminacion:** trabaja en espacio HSV en vez de RGB puro, aplica CLAHE para atenuar diferencias de exposicion, descarta pixeles de sombra/brillo quemado y trata el tono como una magnitud circular (para que rojos cercanos a 0°/360° no se traten como opuestos).
- **Deteccion por region central (ROI):** tanto la webcam como la GUI analizan solo el recuadro central en vez del frame completo, para que el fondo o la mano que sostiene el objeto no contaminen la lectura.
- **Confianza de la prediccion:** el clasificador KNN ahora devuelve, ademas del color, un porcentaje de confianza (voto ponderado por distancia); las lecturas de baja confianza se descartan en vez de mostrarse como si fueran seguras.
- **Enfoque:** se intenta activar el autoenfoque continuo de la camara y, como respaldo, se puede recalibrar el foco manualmente en cualquier momento (tecla `f` en el script de webcam, boton "Reenfocar" en la GUI). Una barra de nitidez en vivo muestra que tan enfocada esta la imagen, y las lecturas se bloquean mientras la imagen esta demasiado borrosa.
- **Interfaz grafica con instrucciones integradas:** la app de escritorio (`color_classification_gui.py`) incluye un boton de Ayuda con la guia completa de uso, ademas de indicaciones visuales (recuadro, barra de nitidez, badge de color) en la propia ventana.

Ver la seccion [Instrucciones de uso](#instrucciones-de-uso) mas arriba para el detalle de como usar cada script.

## Citation
If you use this code for your publications, please cite it as:

    @ONLINE{cr,
        author = "Ahmet Özlü",
        title  = "Color Recognition",
        year   = "2018",
        url    = "https://github.com/ahmetozlu/color_recognition"
    }

## Author
Ahmet Özlü

## License
This system is available under the MIT license. See the LICENSE file for more info.
