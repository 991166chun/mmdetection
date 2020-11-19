# -*- coding: utf-8 -*-
"""
 @File    : grad_cam.py
 @Time    : 2020/3/14 下午4:06
 @Author  : yizuotian
 @Description    :
"""
import cv2
import numpy as np

class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.cam_info = [layer_name,]
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        self.cam_info.append("{}".format([x for x in output.size()]))
        print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        # print(output_grad[0].size())
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, data, index=0):
        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 第几个边框
        :return:
        """
        self.net.zero_grad()
        output = self.net.simple_test(data['img'][0], data['img_metas'][0] )
        
        for i, out in enumerate(output):
            print(out)
            score = out['score']
            label = out['label']
            print(score, label, i)    
            
        score = output[index]['score']
        box = output[index]['bbox']
        proposal_idx = output[index]['ind']
        label = output[index]['label']


        score.backward()
        
        if self.gradient == None:
            return 0,0,0,None
        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]
        
        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU
        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        
        # print(cam)
        orishape = data['img_metas'][0][0]['ori_shape']
        imsize = (orishape[1], orishape[0])
        box = box.detach().numpy().astype(np.int32)
        x1, y1, x2, y2 = box
        cam = cv2.resize(cam, imsize)
        # crop_cam = cam[y1:y2, x1:x2]

        class_id = label.detach().numpy()
        pre_scr = "{:.3f}".format(score.detach().numpy())
        label = self.net.CLASSES[int(class_id)]
        self.cam_info.append(f"label: {label}, score: {pre_scr}, box: {x1}, {y1}, {x2}, {y2}")

        return cam, box, class_id, self.cam_info

class GradCamPlusPlus(GradCAM):
    def __init__(self, net, layer_name):
        super(GradCamPlusPlus, self).__init__(net, layer_name)

    def __call__(self, inputs, index=0):
        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 第几个边框
        :return:
        """
        self.net.zero_grad()
        output = self.net.inference([inputs])
        print(output)
        score = output[0]['instances'].scores[index]
        proposal_idx = output[0]['instances'].indices[index]  # box来自第几个proposal
        score.backward()

        gradient = self.gradient[proposal_idx].cpu().data.numpy()  # [C,H,W]
        gradient = np.maximum(gradient, 0.)  # ReLU
        indicate = np.where(gradient > 0, 1., 0.)  # 示性函数
        norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化
        for i in range(len(norm_factor)):
            norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 避免除零
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]

        weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)

        feature = self.feature[proposal_idx].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to box scale
        box = output[0]['instances'].pred_boxes.tensor[index].detach().numpy().astype(np.int32)
        x1, y1, x2, y2 = box
        cam = cv2.resize(cam, (x2 - x1, y2 - y1))

        return cam

