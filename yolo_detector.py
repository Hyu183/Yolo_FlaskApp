import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from shutil import copyfile
import shutil
import os, glob
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


framework="tf"  #tf, tflite, trt
model="yolov4"  #yolov3 or yolov4
tiny=False      #yolo or yolo-tiny
iou=0.45        #iou threshold
score=0.25      #score threshold
output='./detections/'  #path to output folder
weights_loaded="./checkpoints/yolov4-tf-416"



def yolo_detector(image_path,image_name):
    image_size=416
    imput_image=image_path
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = image_size
    images = [imput_image]
    # load model
    saved_model_loaded = tf.saved_model.load(weights_loaded, tags=[tag_constants.SERVING])


    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
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
        iou_threshold=iou,
        score_threshold=score
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image = utils.draw_bbox(original_image, pred_bbox)
    #cropped_image = utils.draw_bbox(original_image, pred_bbox)
    # image = utils.draw_bbox(image_data*255, pred_bbox)

    #image = Image.fromarray(image.astype(np.uint8))
    #if not FLAGS.dont_show:
        #image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    cv2.imwrite(output + image_name, image)
    return image


# def getUrl(image_path):
#     glass_image=glass_detector(image_path)  
#     image_here="D:\\Glass Final\\detections\\DetectedGlass1.jpg"
#     dst_here="static\\similar_images\\"
#     shutil.copy(image_here, dst_here, follow_symlinks=True)
#     print("Glass image detected............")
#     os.remove("./detections/DetectedGlass1.jpg")
#     return glass_image