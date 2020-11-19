import argparse
import os
import sys
from skimage import io
import argparse
import os
import cv2
import mmcv
import torch
from torch import nn
import numpy as np
import pickle
from mmcv import Config, DictAction
from tools.fuse_conv_bn import fuse_module
from xmTool.grad_cam import GradCAM, GradCamPlusPlus
from mmdet.apis.inference import LoadImage, prepare_data
from mmdet.apis import init_detector, inference_detector

# from mmdet.apis import multi_gpu_test, single_gpu_test
# from mmdet.core import wrap_fp16_model
# from mmdet.datasets import build_dataloader, build_dataset
# from mmdet.models import build_detector

def save_image(image_dicts, input_image_name, output_dir='./result_cam2'):
    prefix = os.path.splitext(input_image_name)[0]
    output_dir = os.path.join(output_dir ,prefix)
    try:
        os.mkdir(output_dir)
    except:
        pass

    for key, image in image_dicts.items():

        ln = key.split('.')
        if len(ln) > 1:
            layer = '{}-{}{}'.format(ln[1], ln[2], ln[3])
        else:
            layer = ln[0]
        io.imsave(os.path.join(output_dir, '{}-{}.jpg'.format(prefix, layer)), image)

def get_last_conv_name(net):
    """
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name

class GuidedBackPropagation(object):

    def __init__(self, net):
        self.net = net
        for (name, module) in self.net.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(self.backward_hook)
        self.net.eval()

    @classmethod
    def backward_hook(cls, module, grad_in, grad_out):
        """
        :param module:
        :param grad_in: tuple,长度为1
        :param grad_out: tuple,长度为1
        :return: tuple(new_grad_in,)
        """
        return torch.clamp(grad_in[0], min=0.0),

    def __call__(self, inputs, index=0):
        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 第几个边框
        :return:
        """
        self.net.zero_grad()
        output = self.net.inference([inputs])
        score = output[0]['instances'].scores[index]
        score.backward()

        return inputs['image'].grad  # [3,H,W]

def norm_image(image):
    """
    normalize image
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def gen_cam(image, mask, cam_info=None):
    """
    generate CAM
    :param image: [H,W,C]
    :param mask: [H,W]
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    
    if cam_info is not None:
        x = int(heatmap.shape[1] * 0.008)
        y = int(heatmap.shape[0] * 0.03)
        y2 = int(heatmap.shape[0] * 0.06)
        y3 = int(heatmap.shape[0] * 0.09)
        print(cam_info)
        font = cv2.FONT_HERSHEY_DUPLEX
        fcolor = (255,255,255)
        lw = 2
        layer = 'layer: ' + cam_info[0]
        pred = cam_info[2]
        # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
        cv2.putText(heatmap, layer, (x, y), font, 1.5, fcolor, lw, cv2.LINE_AA)
        cv2.putText(heatmap, pred, (x, y2), font, 1.5, fcolor, lw, cv2.LINE_AA)

    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    # cam = heatmap + np.float32(image)

    return image, heatmap

def gen_gb(grad):
    """
     guided back propagation
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb

def main(no_img):
    cfg = 'configs/_xm/faster_res50_fpn_voc.py'
    ckpoint = 'work_dirs/faster_res50_fpn_voc/frcnn_final.pth'

    
    img = '/home/r07631006/data/VOCdevkit/VOC2007/JPEGImages/' + no_img + '.jpg'

    # layer_name = 'backbone.layer4.1.conv3'

    with open('xmTool/Grad-CAM.pytorch/detection/layers.txt', 'r') as f:
        layers = f.read().splitlines()
    
    index = 0

    image_dict = {}
    original_image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    cvimg = original_image[..., ::-1]
    image_dict['ori_image'] = cvimg
    while True:
        try:
            for layer_name in layers:
                
                print(layer_name)
                model = init_detector(cfg, ckpoint, device='cpu')
                data = prepare_data(model, img)

                grad_cam = GradCAM(model, layer_name)
                mask, box, class_id, caminfo = grad_cam(data, int(index))  # cam mask
                grad_cam.remove_handlers()
                # print(mask.shape)
                # print(box)
                
                label = model.CLASSES[int(class_id)]
                image_cam, image_dict[ layer_name+ '-'+ str(index)] = gen_cam(cvimg, mask, caminfo)

                label = model.CLASSES[int(class_id)]
                print("label:{}".format(label))

                del model
                del data
                del grad_cam
                del mask, box, class_id
                # torch.cuda.empty_cache()
            index += 1
        except:
            break


    save_image(image_dict, os.path.basename(img))

if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1])