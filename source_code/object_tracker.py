import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import csv
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-person-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/Input_Video5.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'MP4V', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')


class Coord:
    def __init__(self, top, left, bottom, right):
        # actual
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right

        # corners
        self.top_left = (top, left)
        self.top_right = (top, right)
        self.bottom_left = (bottom, left)
        self.bottom_right = (bottom, right)

        # calculate height and width
        self.height = self.bottom - self.top
        self.width = self.right - self.left
        self.area = self.height * self.width

        # mid points
        self.mid_x = self.left + self.width//2
        self.mid_y = self.top + self.height//2 

def get_coord(top, left, bottom, right):
    coord = Coord(top, left, bottom, right)
    return  coord

def is_inside(parent_box, child_box):
    if child_box.left < parent_box.left:
        return False
    
    if child_box.right > parent_box.right:
        return False
    
    if child_box.top < parent_box.top:
        return False

    if child_box.bottom > parent_box.bottom:
        return False

    if child_box.area >= parent_box.area:
        return False
    return True

def mask_person_box_with_face_box(person_boxes, person_labels, face_boxes):
    free_person_boxes = [True]*len(person_boxes)
    person_ids = ['unassigned']*len(person_boxes)
    for x in face_boxes:
        for i in range(len(person_boxes)):
            if not free_person_boxes[i]:
                continue
            else:
                if is_inside(person_boxes[i], x):
                    free_person_boxes[i] = False
                    person_ids[i] = person_labels[i]

    unassigned_faces = [x for x in person_ids if x=='unassigned']
    # if len(unassigned_faces)!=0:
        # print("Alert: %d unassigned faces" % len(unassigned_faces))
    return person_ids

def prepare_details(frame_count, global_person_data, seq_details, person_boxes, person_labels, face_boxes, mask_labels, interval=1):
    person_boxes = [get_coord(x[0], x[1], x[2], x[3]) for x in person_boxes]
    face_boxes = [get_coord(x[0], x[1], x[2], x[3]) for x in face_boxes]
    
    person_ids = mask_person_box_with_face_box(person_boxes, person_labels, face_boxes)

    local_person_data = []
    for x, y, z in zip(person_ids, mask_labels, face_boxes):
        if x=='unassigned':
            continue
        local_person_data.append({'person_id': x.strip().lower(), 'mask_info': y, 'face_roi': z})
    
    for z in local_person_data:
        if z['person_id'] not in global_person_data:
            global_person_data[z['person_id']] = {'mask_state': [], 'time': []}
        
        global_person_data[z['person_id']]['mask_state'].append(z['mask_info'])
        global_person_data[z['person_id']]['time'].append(frame_count)
    
    if frame_count % interval==0:
        masked_faces = [x['face_roi'] for x in local_person_data if x['mask_info'].lower()=='mask']
        non_masked_faces = [x['face_roi'] for x in local_person_data if x['mask_info'].lower()!='mask']

        temp = {'Frame Number': frame_count, 
                'Total non-masked faces': len(non_masked_faces), 
                'Total masked faces': len(masked_faces), 
                'Non-masked Face ROIs': ';'.join([','.join([str(x.mid_x), str(x.mid_y), str(x.width), str(x.height)]) for x in non_masked_faces]),
                'Masked Face ROIs': ';'.join([','.join([str(x.mid_x), str(x.mid_y), str(x.width), str(x.height)]) for x in masked_faces])
                }
        
        seq_details.append(temp)

def write_csv(seq_details):
    if len(seq_details)>0:
        keys = seq_details[0].keys()
        a_file = open("final_result.csv", "w", newline='')
        dict_writer = csv.DictWriter(a_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(seq_details)
        a_file.close()
    else:
        print('no entries are present !!!')

def cal_stats(person_data, fps=16):
    duration = "%.2f" % (len(person_data['time'])/fps)
    start_time = "%.2f" % (person_data['time'][0]/fps)
    end_time = "%.2f" % (person_data['time'][-1]/fps)
    mask_on_duration = "%.2f" % (len([x for x in person_data['mask_state'] if x.lower()=='mask'])/fps) 
    mask_off_duration = "%.2f" % (len([x for x in person_data['mask_state'] if x.lower()!='mask'])/fps)
    return [mask_on_duration, mask_off_duration, start_time, end_time, duration]
        
def write_txt(global_person_data):
    with open('final_result.txt', 'w', encoding='utf-8') as dfile:
        labels = ['person_id', 'mask_on_duration', 'mask_off_duration', 'start_time', 'end_time']
        dfile.write('%s\n' % '\t'.join(labels))
        for person_id, person_data in sorted(global_person_data.items(), key=lambda z: z[0]):
            dfile.write("%s\t%s\n" % (person_id, '\t'.join(cal_stats(person_data)[:-1])))

def final_write(global_person_data, seq_details):
    write_csv(seq_details)
    write_txt(global_person_data)

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = cv2.VideoWriter(FLAGS.output, cv2.VideoWriter_fourcc(*FLAGS.output_format), 30, (640, 480))

    frame_num = 0
    # while video is running

    #OPENCV based mask detection
    Conf_threshold = 0.4
    NMS_threshold = 0.4
    COLORS_cv = [(0, 255, 0), (0, 0, 255), (0, 0, 255),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    class_name_cv = ["Mask", "NoMask", "NoMask"]
    net = cv2.dnn.readNet('mask-yolov4-tiny.weights', 'mask-yolov4-tiny.cfg')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    model_cv = cv2.dnn_DetectionModel(net)
    model_cv.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
    
    global_person_data = {}
    seq_details = []
    frame_count = 0
    while True:
        return_value, frame = vid.read()
        frame = cv2.resize(frame,(640,480))
        classes_cv, scores_cv, boxes_cv = model_cv.detect(frame, Conf_threshold, NMS_threshold)
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        person_boxes = []
        person_labels = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            person_boxes.append([int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])])
            # print([int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])])
            # print('--'*30)
            person_labels.append("%s-%s" % (class_name, str(track.track_id)))

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        # fps = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        face_boxes = []
        mask_labels = []
        for (classid_cv, score_cv, box_cv) in zip(classes_cv, scores_cv, boxes_cv):
            color_cv = COLORS_cv[int(classid_cv) % len(COLORS_cv)]
            label_cv = "%s" % (class_name_cv[classid_cv[0]])
            cv2.rectangle(result, box_cv, color_cv, 1)
            # cv2.rectangle(result, (box_cv[0]-2, box_cv[1]-20),
            #              (box_cv[0]+120, box_cv[1]-4), (100, 130, 100), -1)
            cv2.putText(result, label_cv, (box_cv[0], box_cv[1]-10),
                       cv2.FONT_HERSHEY_COMPLEX, 0.4, color_cv, 1)
            face_boxes.append([int(box_cv[1]), int(box_cv[0]), int(box_cv[1])+int(box_cv[3]), int(box_cv[0])+int(box_cv[2])])
            # print([int(box_cv[1]), int(box_cv[0]), int(box_cv[1])+int(box_cv[3]), int(box_cv[0])+int(box_cv[2])])
            # print('=='*30)
            mask_labels.append(label_cv)


        # fps = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps)
        prepare_details(frame_count, global_person_data, seq_details, person_boxes, person_labels, face_boxes, mask_labels, interval=1)
        frame_count+=1
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
            out.write(result)

        final_write(global_person_data, seq_details)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
