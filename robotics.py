# code for ssd300 object detection

import rospy
from sensor_msgs.msg import CameraInfo, Image
import ros_numpy
import torch
import torchvision
import torch2trt

# ssd object detection for tensorrt integration, takes in 300 by 300 model

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import numpy as np
# get the model, and some labels

precision = 'fp32'
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
# load associated utils for ssd model
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

# throw the model on the gpu
ssd_model = ssd_model.cuda().eval().half()

# get labels for ssd model
classes_to_labels = utils.get_coco_object_dictionary()

# model wrapper for ssd model
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        '''
        given x, a pytorch tensor, run it through the model
        '''
        return self.model(x)

model_w = ModelWrapper(ssd_model).half()

x0 = [np.ones((300, 300, 3))]
# ssd 300 only works for 300 by 300 images
cuda_tensor = utils.prepare_tensor(x0, precision == 'fp16').cuda().half()

# connect torch model to tensorRT
model_trt = torch2trt.torch2trt(model_w, [cuda_tensor], fp16_mode=True)

# normalize image via z score
def normalize(img, mean=128, std=128):
    '''
    given np.array img, mean, and std, perform z-score normalization.
    '''
    res = img - np.full(img.shape, mean)
    final = res / std
    return final

# use gpu
device = torch.device('cuda')

def preprocess(camera_value):
    '''
    
    '''
    global device, normalize
    # resize to fit ssd input dims
    x = cv2.resize(camera_value, (300,300))
    x = normalize(x)
    arr = [x]
    x1 = utils.prepare_tensor(arr, precision == 'fp16')
    return x1


# process the normalized image
def process(camera_value):
    '''
    run the ssd model and calculate detections from output tensor
    '''
    # feed in list of tensor to work with utils function
    # returns that tuple 
    output = model_trt(preprocess(camera_value).half())

    # get output and decode result
    results_per_input = utils.decode_results(output)
    best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]
    cur_img = cv2.resize(camera_value, (300,300))
    for image_idx in range(len(best_results_per_input)):
        bboxes, classes, confidences = best_results_per_input[image_idx]
        for idx in range(len(bboxes)):
            # run through all of the bounding boxes (if there are some)
            left, bot, right, top = bboxes[idx]
            x1, y1, x2, y2 = [val * 300 for val in [left, top, right, bot]]
            # top, left <--> bottom, right
            # calculate area of rectangle of detection
            rect_area = (abs(y1-y2)/300.) * (abs(x1-x2)/300.)
            color = (36,255,12)
            if classes_to_labels[classes[idx] - 1] == 'person' and rect_area >=0.35:
                # have blue color detection if bounding box is of person and fills at least
                # 35% of the screen
                color=(255,0,0)
            cv2.rectangle(cur_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
            cv2.putText(cur_img, classes_to_labels[classes[idx] - 1], (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) 
    # return image with labels as well
    return cur_img


res = np.zeros((480, 856,3))
def vision_callback(data):
    '''
    vision callback for getting data from bebop image topic.
    '''
    global res
    # create np array with correct dtype with ros_numpy package.
    image_arr = ros_numpy.numpify(data)
    x = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
    # process with ssd
    res = process(x)


if __name__ == '__main__':
    # bebop vision detection node
    rospy.init_node('bebop_vision_detection')
    # queue size 1 and large buffer size for images to avoid latency from ssd
    rospy.Subscriber('/bebop/image_raw', Image, vision_callback, queue_size=1, buff_size=2**24)
    while(True):
        # Capture frame-by-frame
        # Display the resulting frame
        cv2.imshow('frame',res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    rospy.spin()