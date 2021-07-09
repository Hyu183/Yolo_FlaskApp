import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl.flags import FLAGS
import core.utils as utils
from tensorflow.python.saved_model import tag_constants

import cv2
import numpy as np


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


framework="tf"  #tf, tflite, trt
model="yolov4"  #yolov3 or yolov4
tiny=False      #yolo or yolo-tiny
iou=0.45        #iou threshold
score=0.25      #score threshold
output='./static/detections/'  #path to output folder
weights_loaded="./checkpoints/yolov4-tf-416"

# load model
def load_yolo_model():
    model = tf.saved_model.load(weights_loaded, tags=[tag_constants.SERVING])
    print('-------------------------------Model Yolov4 Loaded-------------------------------')
    return model

def yolo_detector(image_path,image_name,model):
    image_size=416
    imput_image=image_path
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = image_size
    images = [imput_image]
    

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    infer = model.signatures['serving_default']
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
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    cv2.imwrite(output + image_name, image)
    return image