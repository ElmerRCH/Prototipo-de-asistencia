from tkinter import *
from tkinter.messagebox import *
import cv2
import os
import imutils
import numpy as np

def EncenderCamara():
   
    capture = cv2.VideoCapture(0)

    while (capture.isOpened()):
        ret, frame = capture.read()
        cv2.imshow('webCam',frame)
        capture.release()
        showinfo("alert","camara funcionando")
        if (cv2.waitKey(6) == ord('s')):
            break

    capture.release()
    cv2.destroyAllWindows()
    
def capturar():
    personName = 'Elmer'
    dataPath = 'D:/Proyectito'#Cambia a la ruta donde hayas almacenado Data
    personPath = dataPath + '/' + personName
    if not os.path.exists(personPath):
        print('Carpeta creada: ',personPath)
        os.makedirs(personPath)
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    #cap = cv2.VideoCapture('Video.mp4')
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    count = 0
    while True:
        
        ret, frame = cap.read()
        if ret == False: break
        frame =  imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()
        faces = faceClassif.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro)
            count = count + 1
        cv2.imshow('frame',frame)
        k =  cv2.waitKey(1)
        if k == 27 or count >= 300:
            break
    cap.release()
    cv2.destroyAllWindows()

def entrenar():
   

    dataPath1 = 'D:/Proyectito/Elmer'
    dataPath = 'D:/Proyectito'#Cambia a la ruta donde hayas almacenado Data
    peopleList = os.listdir(dataPath)
    print('Lista de personas: ', peopleList)

    labels = []
    facesData = []
    label = 0

    for nameDir in peopleList:
        personPath = dataPath + '/' + nameDir
        print('Leyendo las imágenes')

        for fileName in os.listdir(dataPath1):
            print('Rostros: ', nameDir + '/' + fileName)
            labels.append(label)
            facesData.append(cv2.imread(personPath+'/'+fileName,0))
            #image = cv2.imread(personPath+'/'+fileName,0)
            #cv2.imshow('image',image)
            #cv2.waitKey(10)
        label = label + 1


    #print('labels= ',labels)
    #print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
    #print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))

    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    #face_recognizer = cv2.face.FisherFaceRecognizer_create()
    #face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("entrenando....")
    face_recognizer.train(facesData,np.array(labels))

    face_recognizer.write('modeloEigenFace.xml')
    print("almacenado terminado")

def EnableSystem():
    dataPath = 'D:/Proyectito' #Cambia a la ruta donde hayas almacenado Data
    imagePaths = os.listdir(dataPath)
    print('imagePaths=',imagePaths)

    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    #face_recognizer = cv2.face.FisherFaceRecognizer_create()
    #face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Leyendo el modelo
    face_recognizer.read('modeloEigenFace.xml')
    #face_recognizer.read('modeloFisherFace.xml')
    #face_recognizer.read('modeloLBPHFace.xml')

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    #cap = cv2.VideoCapture('Video.mp4')

    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    while True:
        ret,frame = cap.read()
        if ret == False: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
            
            # EigenFaces
            if result[0] < 3500:
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            elif result[1] :
                cv2.putText(frame,'{}'.format(imagePaths[result[1]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            else:
                cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
            """  # FisherFace
            if result[1] < 500:
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            else:
                cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
            
            # LBPHFace
            if result[1] < 70:
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            else:
                cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)  """
            
        cv2.imshow('frame',frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



def cerrar():
 layout.destroy() 


layout =Tk()
layout.geometry("1920x1080")
layout.title("Gestor de bases de datos")
menus=Menu(layout)
layout.config(menu=menus,bg="gray")
#------------MENUS----------------
boton = Button(layout,text = 'Probar camara',bg = '#f20033',command =EncenderCamara)
boton.pack()
boton.place(x = 200, y = 200)
boton.config(width = "30", height = "25")
#=========================================================================================================

boton1 = Button(layout,text = 'Cerrar programa',bg = '#f20039',command = cerrar)
boton1.pack ()
boton1.place(x = 412, y = 590)
boton1.config(width = "100", height = "2")

boton2 = Button(layout,text = 'Capturar rostros',bg = '#f20039',command = capturar)
boton2.pack ()
boton2.place(x = 1050, y = 250 )
boton2.config(width = "15", height = "3")

boton3 = Button(layout,text = 'Entrenar rostros',bg = '#f20039',command = entrenar)
boton3.pack ()
boton3.place(x = 1050, y = 350)
boton3.config(width = "15", height = "3")


boton4 = Button(layout,text = 'Ver Camara',bg = '#f20039',command = EnableSystem)
boton4.pack ()
boton4.place(x = 1050, y = 450)
boton4.config(width = "15", height = "3")

layout.mainloop()

