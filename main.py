import cv2

config_model = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt.txt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model,config_model)

class_labels = []
file_name = 'labels.txt'
with open(file_name,'r') as i:
    class_labels = i.read().rstrip('\n').split('\n')

#model input parameters
model.setInputSize(340,340)
model.setInputScale(1.0 /127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

font_scale= 2
font = cv2.FONT_HERSHEY_PLAIN

# Uploading an file/webcam
cap = cv2.VideoCapture('SampleVideok1.mp4')
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cant Open the file or the webcam")

while True:
  ret,frame = cap.read()
  ClassIndex , confidence , bbox = model.detect(frame, confThreshold = 0.45)
  print(ClassIndex)
  if(len(ClassIndex) != 0):
    for ClassInd, conf , boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
      if(ClassInd <=80):
        cv2.rectangle(frame,boxes,(0,0,255),4)
        cv2.putText(frame,class_labels[ClassInd-1],(boxes[0]+20,boxes[1]+80),font,fontScale=font_scale,color=(0,255,0),thickness=2)
  cv2.imshow('Object Detection Program',frame)
  if cv2.waitKey(2) & 0xff == ord('q'):
    break


cap.release()
cv2.destroyAllWindows()






