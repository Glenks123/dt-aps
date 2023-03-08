try:
    import cv2
    import pyfirmata
except:
    import pip
    packages = ['cv2', 'pyfirmata']
    for package in packages:
        pip.main(['install', 'package'])

    import cv2
    import pyfirmata


# Initiate communicatino with Arduino
board = pyfirmata.Arduino('COM3')
print('Communication Successfully started')

PIEZO_PIN = board.get_pin('d:11:p')


thres = 0.7

##img = cv2.imread('face.png')
cap = cv2.VideoCapture(2)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

classNames = []
classFile = 'coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()

    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)  # green
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10,
                        box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            if classId == 77:
                PIEZO_PIN.write(0.9)
                board.pass_time(1)
                PIEZO_PIN.write(0)
                board.pass_time(0.5)

    cv2.imshow('Output', img)
    cv2.waitKey(1)
