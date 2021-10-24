import cv2 as cv
import time
Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (0, 0, 255),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

class_name = ["Mask", "NoMask", "NoMask"]
#with open('classes.txt', 'r') as f:
#    class_name = [cname.strip() for cname in f.readlines()]
# print(class_name)
net = cv.dnn.readNet('mask-yolov4-tiny.weights', 'mask-yolov4-tiny.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


cap = cv.VideoCapture(0)
frame_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
while True:
    ret, frame = cap.read()
    #frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_name[classid[0]], score)
        cv.rectangle(frame, box, color, 1)
        # cv.line(frame, (box[0]-3, box[1]-15),
        #         (box[0]+110, box[1]-15), (0, 0, 0), 15)
        cv.putText(frame, label, (box[0], box[1]-10),
                   cv.FONT_HERSHEY_COMPLEX, 0.4, color, 1)
    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
