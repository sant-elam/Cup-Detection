
import cv2 as cv
import mediapipe as mp

holistic = mp.solutions.holistic
drawing_utils = mp.solutions.drawing_utils

drwLandmark = drawing_utils.DrawingSpec((200, 50, 50), thickness=1, circle_radius=2)
drwConnections = drawing_utils.DrawingSpec((50, 200, 50), thickness=1)

path = 'C:/MEDIA_PIPE/Holistic/SL_MO_VID_20191029_183206.mp4'
cap = cv.VideoCapture(path)

model = holistic.Holistic(static_image_mode=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.3)

i=0
while True:
    
    result, image_org = cap.read()
    
 
    if result:
        image = cv.cvtColor(image_org, cv.COLOR_BGR2RGB)
        
        output = model.process(image)
        
        if output.face_landmarks:
                
                '''
                drawing_utils.draw_landmarks(image = image_org, 
                                             landmark_list = output.face_landmarks,
                                             connections = holistic.FACEMESH_TESSELATION,
                                             landmark_drawing_spec= drwLandmark,
                                             connection_drawing_spec=drwConnections)
                '''  
                drawing_utils.draw_landmarks(image = image_org, 
                                             landmark_list = output.pose_landmarks,
                                             connections = holistic.POSE_CONNECTIONS,
                                             landmark_drawing_spec= drwLandmark,
                                             connection_drawing_spec=drwConnections)
                             
        
        cv.imshow("Holistic", image_org)
        if cv.waitKey(1) & 255 == 27:
                break
    else:
        break
    
cv.destroyAllWindows()
cap.release()