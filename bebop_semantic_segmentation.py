
import rospy
from sensor_msgs.msg import CameraInfo, Image
import ros_numpy
# real time semantic segmentation model with opencv with bebop camera.
import torch
import torchvision
import torch2trt

# deeplabv3 semantic segmentation model
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)

# throw the model onto gpu
model = model.cuda().eval().half()

# model wrapper
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        # need to get 'out' for actual output
        return self.model(x)['out']

model_w = ModelWrapper(model).half()

# sample data for getting the model
data = torch.ones((1, 3, 224, 224)).cuda().half()

# convert to tensorRT
model_trt = torch2trt.torch2trt(model_w, [data], fp16_mode=True)

# was opencv issue, need to do this to not get import cv2 error
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


import cv2
import numpy as np

device = torch.device('cuda')
# assigned means and standard deviations
mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])
# normalizing effect
normalize = torchvision.transforms.Normalize(mean, stdev)

# pre-process image, set to right image format, transpose, normalize
def preprocess(camera_value):
    global device, normalize
    x = cv2.resize(camera_value, (224,224))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2,0,1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x

# run model and create mask image for objects
def process(camera_value):
    output = model_trt(preprocess(camera_value).half())[0].detach()
    output = output.cpu().float().numpy()
    mask = 1.0 * (output.argmax(0) == 15)
    res = mask[:, :, None] * cv2.resize(camera_value, (224,224))
    return res

res = np.zeros((480, 856,3))
def vision_callback(data):
    '''
    vision callback for getting data from bebop image topic.
    '''
    global res
    # create np array with correct dtype with ros_numpy package.
    image_arr = ros_numpy.numpify(data)
    # process with deeplab semantic segmentation
    res = process(image_arr)


if __name__ == '__main__':
    # bebop vision detection node
    rospy.init_node('bebop_vision_segmentation')
    # queue size 1 and large buffer size for images to avoid latency from ssd
    rospy.Subscriber('/bebop/image_raw', Image, vision_callback, queue_size=1, buff_size=2**24)
    while(True):
        # Capture frame-by-frame
        # Display the resulting frame
        cv2.imshow('frame',res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    rospy.spin()