import os
import torch
from PIL import Image
from donut import DonutModel

class GlobalModel:
    def __init__(self, global_model_path, empty_folder, global_limit_token):
        self.global_model_path = global_model_path
        self.empty_folder = empty_folder
        self.global_limit_token = global_limit_token
        self.global_model = self.load_model()

    def load_model(self):
        global_model = DonutModel.from_pretrained(self.global_model_path)
        if torch.cuda.is_available():
            global_model.half()
            device = torch.device("cuda:0")
            global_model.to(device)
        return global_model

    def process_image(self, image):
        try:
            image1 = Image.fromarray(image)
            task_name = os.path.basename(self.empty_folder)
            result = self.global_model.inference(image=image1, prompt=f"<s_{task_name}>")["predictions"][0]

            if 'c2t' in result:
                text_result = result['c2t']
            else:
                text_result = result['text_sequence']

            limited_result = ' '.join(text_result.split()[:self.global_limit_token])
            return limited_result
        except Exception as e:
            return ""
