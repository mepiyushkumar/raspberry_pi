import cv2
import time
import websocket
import json

CONFIDENCE_THRESHOLD = 0.8
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("./classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

vc = cv2.VideoCapture(0)

net = cv2.dnn.readNet("./custom-yolov4-tiny-detector_best.weights", "./yolov4-custom-uday.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255, swapRB=True)

uri = "wss://gtg3p8yh66.execute-api.us-east-1.amazonaws.com/production/"
ws = websocket.create_connection(uri)

last_detection_time = time.time()

while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        exit()

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    start_drawing = time.time()
    detected_class = ""
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if class_names[classid] in ["Bread","Cardboard","Metal Can","Plastic Bag"]:
            detected_class = class_names[classid]
    end_drawing = time.time()

    if detected_class:
        if time.time() - last_detection_time >= 10:
            ws.send(json.dumps(detected_class))
            last_detection_time = time.time()
            time.sleep(10)
        detected_class = ""
    
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("detections", frame)
