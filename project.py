from imageai.Detection import ObjectDetection
import os
import cv2
import pyttsx3
import engineio

engineio = pyttsx3.init()
voices = engineio.getProperty('voices')
engineio.setProperty('rate', 150)
engineio.setProperty('voice',voices[0].id)

def speak(text):
    engineio.say(text)
    engineio.runAndWait()

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)
    

    if k == 27:
        # ESC pressed
        print("Detecting objects...")
        break
    elif k == 32:
        # SPACE pressed
        img_name="test.png"
        cv2.imwrite(img_name, frame)
        print("image token")
        
        
cam.release()
cv2.destroyAllWindows()

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , 'test.png'), output_image_path=os.path.join(execution_path , "imagenew.png"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
    speak(eachObject["name"])











