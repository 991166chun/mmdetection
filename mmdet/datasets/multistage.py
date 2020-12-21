import mmcv
import numpy as np

from mmdet.core import eval_map
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MultiStageDataset(CustomDataset):

    CLASSES_1 = ('disease','back')

    CLASSES_2 = ('brownblight', 'blister', 'algal',  'fungi_early',
                 'miner',  'thrips',
                 'mosquito_early', 'mosquito_late',
                 'moth', 'tortrix',   'flushworm',
                 'roller')

    def __init__(self,**kwargs):
        super(XMLDataset, self).__init__(**kwargs)
        self.cat2category = {cat: i for i, cat in enumerate(self.CLASSES_1)}
        self.cat2pest = {cat: i for i, cat in enumerate(self.CLASSES_2)}
   
    def load_annotations(self, ann_file):
        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = f'JPEGImages/{img_id}.jpg'
            xml_path = osp.join(self.img_prefix, 'MultiLabel',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = 0
            height = 0
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, 'JPEGImages',
                                    '{}.jpg'.format(img_id))
                img = Image.open(img_path)
                width, height = img.size
            data_infos.append(
                dict(id=img_id,
                     filename=filename, 
                     width=width, 
                     height=height))

        return data_infos

    def get_subset_by_classes(self):
        """Filter imgs by user-defined categories
        """
        subset_data_infos = []
        for data_info in self.data_infos:
            img_id = data_info['id']
            xml_path = osp.join(self.img_prefix, 'MultiLabel',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                
                # category = obj.find('category').text
                # if category in self.CLASSES_1:
                #     subset_data_infos.append(data_info)
                #     break

                label = obj.find('label').text
                if label in self.CLASSES_2:
                    subset_data_infos.append(data_info)
                    break
                

        return subset_data_infos

    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'MultiLabel', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        cates = []
        pests = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):


            category = obj.find('category').text
            if category not in self.CLASSES_1:
                continue
            category = self.cat2category[category]

            pest = obj.find('label').text
            if pest not in self.CLASSES_2:
                continue
            pest = self.cat2pest[pest]

            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False
            
            bboxes.append(bbox)
            cates.append(category)
            pests.append(pest)

        
        bboxes = np.array(bboxes, ndmin=2) - 1
        cates = np.array(cates)
        pests = np.array(pests)

        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))

        ann = dict(
            bboxes=bboxes.astype(np.float32),
            category=cates.astype(np.int64),
            pests=pests.astype(np.int64),

            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP',]
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i) for i in range(len(self))]

        eval_results = {}

        if metric == 'mAP':
            assert isinstance(iou_thr, float)

            for stage in range(3):

                if stage == 0:
                    ds_name = self.dataset.CLASSES_1
                    label = 'category'
                else:
                    ds_name = self.dataset.CLASSES_2
                    label = 'pests'

                stage_annotations = []
                for anno in annotations:
                    anno['labels']=anno[label]
                    stage_annotations.append(anno)
                '''
                result (temp) : [[det_results_1], [det_results_2], [det_results_3]]
                det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
                    The outer list indicates images, and the inner list indicates
                    per-class detected bboxes.
                annotations (list[dict]): Ground truth annotations where each item of
                    the list indicates an image. Keys of annotations are:

                - `bboxes`: numpy array of shape (n, 4)
                - `labels`: numpy array of shape (n, )
                - `bboxes_ignore` (optional): numpy array of shape (k, 4)
                - `labels_ignore` (optional): numpy array of shape (k, )
        
                '''
                print('eval stage %d mAP from outputs' %(stage+1))
                mean_ap, _ = eval_map(
                    results[stage],
                    stage_annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger=logger)
                eval_results['mAP'] = mean_ap
 
        return eval_results