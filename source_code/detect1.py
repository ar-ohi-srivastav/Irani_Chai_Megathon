import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import face_recognition
from numpy import dot
from numpy.linalg import norm


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='cfg/yolo-fastest.cfg', help='*.cfg path')
parser.add_argument('--names', type=str, default='data/face_mask.names', help='*.names path')
parser.add_argument('--weights', type=str, default='weights/best.weights', help='weights path')
parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
opt = parser.parse_args()


def cosine_sim(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

def dist_compare(val, threshold, distance='euclid'):
    if distance=='euclid':
        return val<threshold
    else:
        return val>threshold

def identify_and_tag_faces(face_emb, new_embs, threshold=.6, distance='euclid'):
    person_id = []
    # for x in face_emb:
    #     print(x)
    # print('--'*30)
    if len(face_emb)==0:
        # add new faces to embedding
        for i, x in enumerate(new_embs):
            face_emb['person_%s' % str(i+1)] = {'emb': x}
            person_id.append('person_%s' % str(i+1))
    else:
        last_person_id = -1
        for k in face_emb:
            face_emb[k]['flag'] = True
            last_person_id = max(int(k.split('_')[1]), last_person_id)

        new_face_embds = []
        for x in new_embs:
            temp_face_emb = {k:v['emb'] for k, v in face_emb.items() if v.get('flag', None)==True}
            cand_face_embedding = [(k, v) for k, v in sorted(temp_face_emb.items(), key=lambda z: z[0])]
            
            if distance=='euclid':
                res = face_recognition.face_distance([z[1] for z in cand_face_embedding], x)      
                final_res = [(cand_face_embedding[i][0], res[i]) for i in range(len(res))]
                final_res = sorted(final_res, key=lambda z: z[1])
            else:
                res = [cosine_sim(z[1], x) for z in cand_face_embedding]
                final_res = [(cand_face_embedding[i][0], res[i]) for i in range(len(res))]
                final_res = sorted(final_res, key=lambda z: z[1], reverse=True)
            # add if only if there
            if len(final_res)>0 and dist_compare(final_res[0][1], threshold, distance=distance): #final_res[0][1]<threshold:
                person_id.append(final_res[0])
                face_emb[final_res[0][0]]['flag']=False
            else:
                last_person_id+=1
                face_emb["person_%s" % str(last_person_id)] = {'emb': x}
                person_id.append("person_%s" % str(last_person_id))
    
    return person_id

def identify_and_tag_faces_v2(frame_id, previous_person_ids, face_emb, new_embs, threshold=.6, distance='euclid', interval=15):
    person_ids = []
    # for x in face_emb:
    #     print(x)
    # print('--'*30)
    if frame_id % interval != 0:
        return previous_person_ids

    if len(face_emb)==0:
        # add new faces to embedding
        for i, x in enumerate(new_embs):
            face_emb['person_%s' % str(i+1)] = {'emb': x}
            person_ids.append('person_%s' % str(i+1))
    else:
        last_person_id = -1
        for k in face_emb:
            face_emb[k]['flag'] = True
            last_person_id = max(int(k.split('_')[1]), last_person_id)

        # running over m * n
        temp_face_emb = {k:v['emb'] for k, v in face_emb.items() if v.get('flag', None)==True}
        cand_face_embedding = [(k, v) for k, v in sorted(temp_face_emb.items(), key=lambda z: z[0])]
        final_dist = []
        for i, x in enumerate(new_embs):
            for z in cand_face_embedding:
                if distance=='euclid':
                    res = face_recognition.face_distance(z[1], x)      
                else:
                    res = cosine_sim(z[1], x)
                final_dist.append([i, z[0], res])

        if distance=='euclid':
            final_dist = sorted(final_dist, key=lambda x: x[-1])
        else:
            final_dist = sorted(final_dist, key=lambda x: x[-1], reverse=True)

        new_cand_done = set()
        person_identified = set()
        person_ids = ['unassigned']*len(new_embs)

        for z in final:
            if not dist_compare(z[-1], threshold, distance=distance):
                continue
            # check if person is already identified
            if z[0] in new_cand_done:
                continue
            # check if person id is already used
            if z[1] in person_identified:
                continue
            person_ids[z[0]] = z[1]
            new_cand_done.add(z[0])
            person_identified.add(z[1])
        
        # check unidentified person and assign new label
        for idx in range(len(new_embs)):
            if person_ids[idx]=='unassigned':
                last_person_id+=1
                face_emb["person_%s" % last_person_id] = new_embs[idx]
                person_ids[idx] = "person_%s" % last_person_id
    
    return person_ids

def update_person_text_data(frame_id, local_person_data, person_text_data, interval=15):
    if not frame_id%interval==0:
        return person_text_data
    
    for x in person_text_data:
        if x in [z[0] for z in local_person_data]:
        
        person_id = x[0]
        mask_info = x[1]
        if person_id in person_text_data:
            if mask_info=='Mask':

        else:
            person_text_data[person_id] = {'mask': [mask_info], 'entry_time': [frame_id//interval], 'exit_time': []}




    # update the details


def detect(save_img=True, out="static/images/test.jpg", source='data/samples/good_test.jpg'):
    person_text_data = {}
    
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    weights, half, view_img, save_txt = opt.weights, opt.half, True, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors = [[0, 69, 255], [170, 178, 32]] # Color for binary classification in B,G,R

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    frame_count = -1
    for path, img, im0s, vid_cap in dataset:
        frame_count+=1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        label = None
        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            #save_path = str(Path(out) / Path(p).name)
            save_path = str(Path(out) / 'test.jpg')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
#                for c in det[:, -1].unique():
#                    n = (det[:, -1] == c).sum()  # detections per class
#                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # person re-identification
                bounding_boxes = []
                for *xyxy, conf in det:
                    mtop, mleft, mbotton, mright = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    bounding_boxes.append((mtop, mright, mbotton, mleft))
                
                encodings = face_recognition.face_encodings(image, bounding_boxes)
                person_ids = identify_and_tag_faces(frame_count, person_map, encodings)

                collected_data = []
                # Write results
                for person_id, *xyxy, conf, cls in zip(person_ids, det):

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if True: #save_img or view_img:  # Add bbox to image
                        label = '%s | %s %.2f' % (person_id, names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                    collected_data.append([person_id, names[int(cls)]])
            # updating the person information
            update_person_text_data(frame_count, collected_data)

            # Print time (inference + NMS)
            if label:
                print('%sDone. (%.3fs)' % (s, t2 - t1))
                print(label)
            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                # if cv2.waitKey(1) == ord('q'):  # q to quit
                #     raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    return label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolo-fastest.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/face_mask.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/best.weights', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)

    with torch.no_grad():
        detect(out=opt.output, source=opt.source, save_img=False)
