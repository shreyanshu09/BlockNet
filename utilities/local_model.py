import os
import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

class LocalModel:
    def __init__(self, weights_path, yaml_file):
        self.model = self.load_model(weights_path, yaml_file)

    def load_model(self, weights_path, yaml_file):
        device = select_device('cpu')
        model = DetectMultiBackend(weights_path, device=device, dnn=False, data=yaml_file, fp16=False)
        return model

    def process_image(self, img_path, stride, names, pt):
        dataset = LoadImages(img_path, img_size=(640, 640), stride=stride, auto=pt)
        for path, im, im0s, _, _ in dataset:
            im = torch.from_numpy(im).to(self.model.device)
            pred = self.model(im, augment=True, visualize=False)
            pred = non_max_suppression(pred, 0.35, 0.7, max_det=100)
            sorted_data_list = []
            for i, det in enumerate(pred):
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    sorted_data_list.append((int(cls), xywh))
        return tuple(sorted_data_list)
