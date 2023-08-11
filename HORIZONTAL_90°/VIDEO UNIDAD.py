import pandas as pd
import datetime
from shutil import rmtree
import mediapipe as mp
import numpy as np
import time
import cv2
import os
import glob

start_time = time.perf_counter()

#CREA UNA MATRIZ PARA LOS ANGULOS DE LAS FALANGES

anguloD = np.zeros((200000,10))
anguloM = np.zeros((200000,8))
anguloP = np.zeros((200000,10))

#MATRICES FALANGES

colD = 0
filD = 0
coluD = 0
filaD = 0

colM = 0
filM = 0
coluM = 0
filaM = 0

colP = 0
filP = 0
coluP = 0
filaP = 0

#FALANGES A ANALIZAR

Distal = [[4,3,2], [8,7,6], [12,11,10], [16,15,14], [20,19,18]]
Medial = [[7,6,5], [11,10,9], [15,14,13], [19,18,17]]
Proximal = [[3,2,1], [6,5,0], [10,9,0], [14,13,0], [18,17,0]]

#TODAS LAS FALANGES

joint_list = [[8,7,6], [12,11,10], [16,15,14], [20,19,18], 
              [4,3,2], [7,6,5], [11,10,9], [15,14,13], [19,18,17],
              [3,2,1], [6,5,0], [10,9,0], [14,13,0], [18,17,0]]

prb = 0
ind = 0
colu = 0
fila = 0

#DETECTAR LA CANTIDAD DE VIDEOS Y SUS NOMBRES

video = "2-3.mp4"  

if video.endswith(('.mp4', '.avi', '.mkv')):
    video = os.path.splitext(video)[0]

#CREA UNA CARPETA POR PARTICIPANTE
    
if os.path.exists(video):
    rmtree(video)

parent_path = os.path.join(os.getcwd(), video)
os.makedirs(parent_path)

#ANÁLISIS CINEMÁTICO DE CADA VIDEO AUTOMÁTICO

anguloD = np.zeros((200000,10))
anguloM = np.zeros((200000,8))
anguloP = np.zeros((200000,10))

dedo = 1
n_frame = 0

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(video+'.mp4')
now = datetime.datetime.now()
print(video + ": " + now.strftime("%H:%M:%S"))

def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            #Procesa resultados
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            #Extraer coordenadas
            coords  = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [width, heigh]).astype(int))
            output = text, coords
    return output

with mp_hands.Hands(static_image_mode = False,min_detection_confidence = 0.5, min_tracking_confidence = 0.5, max_num_hands = 2) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if frame is None:
            break
        
        #frame = cv2.flip(frame,1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame.flags.writeable = False
        results = hands.process(frame_rgb)
        
        heigh, width, _ = frame.shape
        
        frame=cv2.resize(frame,(1200,600))
        
        heigh, width, _ = frame.shape
        
        nombre_frame = f"frame{n_frame}.jpg"
        
       
            
        #Dibujar el valor de los angulos en tiempo real
        #draw_finger_angles(frame, results, joint_list)
        #cv2.imshow('Hand Tracking', frame)
        
        for hand in results.multi_hand_landmarks:
            
            #LOOP FALANGES DISTAL
            for joint in Distal:
                a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) #1ra coordenada
                b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) #2da coordenada
                c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) #3ra coordenada
    
                radians = np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
                angle = np.abs(radians*180.0/np.pi)
                if angle > 180.0:
                    angle = 360-angle
                    
                anguloD[filD,colD] = round(angle,2)
                
                cv2.putText(frame, str(round(angle, 2)), tuple(np.multiply(b, [width, heigh]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                colD = colD+1
                if colD == 10:
                    colD = 0
                    filD = filD+1
            
            for joint in Medial:
                a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) #1ra coordenada
                b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) #2da coordenada
                c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) #3ra coordenada
    
                radians = np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
                angle = np.abs(radians*180.0/np.pi)
                if angle > 180.0:
                    angle = 360-angle
                    
                anguloM[filM,colM] = round(angle,2)
                
                cv2.putText(frame, str(round(angle, 2)), tuple(np.multiply(b, [width, heigh]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                colM = colM+1
                if colM == 8:
                    colM = 0
                    filM = filM+1
                    
            for joint in Proximal:
                a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) #1ra coordenada
                b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) #2da coordenada
                c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) #3ra coordenada
    
                radians = np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
                angle = np.abs(radians*180.0/np.pi)
                if angle > 180.0:
                    angle = 360-angle
                    
                anguloP[filP,colP] = round(angle,2)
                
                cv2.putText(frame, str(round(angle, 2)), tuple(np.multiply(b, [width, heigh]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                colP = colP+1
                if colP == 10:
                    colP = 0
                    filP = filP+1
        
        #cv2.imwrite(os.path.join(video,'{}.jpg'.format(uuid.uuid1())),frame)
        cv2.imwrite(os.path.join(parent_path, nombre_frame),frame)
        n_frame += 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()       
 
#CREAR EL DATAFRAME CON LOS ÁNGULOS

angD_dtf = pd.DataFrame(anguloD,
                         columns = ['Pulgar Der', 'Indice Der',
                                    'Medio Der', 'Anular Der',
                                    'Meñique Der', 'Pulgar Izq',
                                    'Indice Izq', 'Medio Izq',
                                    'Anular Izq', 'Meñique Izq'])

angM_dtf = pd.DataFrame(anguloM,
                         columns = ['Indice Der', 'Medio Der', 
                                    'Anular Der', 'Meñique Der', 
                                    'Indice Izq', 'Medio Izq',
                                    'Anular Izq', 'Meñique Izq'])


angP_dtf = pd.DataFrame(anguloP,
                         columns = ['Pulgar Der', 'Indice Der',
                                    'Medio Der', 'Anular Der',
                                    'Meñique Der', 'Pulgar Izq',
                                    'Indice Izq', 'Medio Izq',
                                    'Anular Izq', 'Meñique Izq'])

angD_dtf = angD_dtf[(angD_dtf != 0).all(axis=1)]         #ELIMINA FILAS CON 0
angD_dtf.to_excel(os.path.join(parent_path, 'Angulos Distal.xlsx'), index=False)      #CREA UN .CVS

angM_dtf = angM_dtf[(angM_dtf != 0).all(axis=1)] 
angM_dtf.to_excel(os.path.join(parent_path, 'Angulos Medial.xlsx'), index=False)

angP_dtf = angP_dtf[(angP_dtf != 0).all(axis=1)] 
angP_dtf.to_excel(os.path.join(parent_path, 'Angulos Proximal.xlsx'), index=False)

end_time = time.perf_counter()
run_time = end_time - start_time
print(f"El código tardó {run_time:.5f} segundos en ejecutarse.")