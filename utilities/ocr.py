import os
import sys
import numpy as np
import cv2
import json
import re
import numpy as np

sys.path.append("ocr_pororo")
from main import PororoOcr

class OCR:
    def __init__(self):
        self.ocr = PororoOcr()

    def run_ocr(self, img):
        self.ocr.run_ocr(img, debug=False)

    def get_ocr_result(self):
        return self.ocr.get_ocr_result()

    def pororo_ocr(self, img_path):
        self.run_ocr(img_path)
        res = self.get_ocr_result()
        word_coordinates = []
        for i in range(len(res['description'])):
            word = res['description'][i]
            vertices = res['bounding_poly'][i]['vertices']
            x_min = min(vertex['x'] for vertex in vertices)
            y_min = min(vertex['y'] for vertex in vertices)
            x_max = max(vertex['x'] for vertex in vertices)
            y_max = max(vertex['y'] for vertex in vertices)
            width = x_max - x_min
            height = y_max - y_min
            word_coordinates.append((word, (x_min, y_min, width, height)))
        return word_coordinates
