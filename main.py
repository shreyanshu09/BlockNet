# python main.py image.png --task "Short Description" --lang Korean --api_key "YOUR_API_KEY_HERE"


import argparse
import subprocess
import numpy as np
import base64
import io
import tempfile
import cv2
from PIL import Image
from global_model import GlobalModel
from local_model import LocalModel
from ocr import OCR
from integration_gpt4 import IntegrationGPT4

class MainController:
    def __init__(self, api_key, global_model, local_model, ocr, integration_gpt4):
        self.api_key = api_key
        self.global_model = global_model
        self.local_model = local_model
        self.ocr = ocr
        self.integration_gpt4 = integration_gpt4

    def process_image(self, image_path, task="Short Description", lang="Korean", question=None):
        input_image = Image.open(image_path)
        img = np.array(input_image)

        # OCR
        ocr_output = self.ocr.pororo_ocr(img)
        
        # Calling global model
        global_model_output = self.global_model.process_image(input_image)

        # Processing Local model result
        txt_data_labels = self.local_model.process_image(image_path, stride=0, names=[], pt=False)
        img = np.array(input_image)
        edge, node = self.local_model.read_list(txt_data_labels, img)
        crop_image_result = self.local_model.find_closest_node(edge, node)
        json_result = {'results': crop_image_result}
        local_model_output, ocr_output = self.local_model.process_json_file(img, json_result)
        
        # Encoding image
        img_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        pil_image2 = Image.fromarray(np.uint8(img_rgb))
        image_bytes = io.BytesIO()
        pil_image2.save(image_bytes, format='JPEG')
        image_data = image_bytes.getvalue()
        encode_image = base64.b64encode(image_data).decode('utf-8')
        
        if task == "Short QA":
            final_result = self.integration_gpt4.answer_question(self.api_key, encode_image, local_model_output, global_model_output, ocr_output, lang, question, todo='very shortly')
        elif task == "Long QA":
            final_result = self.integration_gpt4.answer_question(self.api_key, encode_image, local_model_output, global_model_output, ocr_output, lang, question, todo='in detail')
        elif task == "Short Description":
            final_result = self.integration_gpt4.generate_answer(self.api_key, encode_image, local_model_output, global_model_output, ocr_output, lang, todo='a very short Description in one paragraph only')
        else:
            final_result = self.integration_gpt4.generate_answer(self.api_key, encode_image, local_model_output, global_model_output, ocr_output, lang, todo='the Description in detail')
        return final_result

def main():
    parser = argparse.ArgumentParser(description='Process block diagram images and generate descriptions or answer questions.')
    parser.add_argument('image_path', type=str, help='Path to the input image file.')
    parser.add_argument('--task', type=str, default='Short Description', choices=['Short Description', 'Long Description', 'Short QA', 'Long QA'], help='Task to perform: "Short Description", "Long Description", "Short QA", or "Long QA" (default: "Short Description").')
    parser.add_argument('--lang', type=str, default='Korean', choices=['Korean', 'English'], help='Language for the generated text: "Korean" or "English" (default: "Korean").')
    parser.add_argument('--question', type=str, default=None, help='Optional question to answer (only applicable for QA tasks).')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key.')
    args = parser.parse_args()

    # Initialize global_model, local_model, ocr, and integration_gpt4
    global_model = GlobalModel()
    local_model = LocalModel()
    ocr = OCR()
    integration_gpt4 = IntegrationGPT4()

    # Initialize the Main Controller with necessary dependencies
    main_controller = MainController(args.api_key, global_model, local_model, ocr, integration_gpt4)

    # Process the image and get the final result
    final_result = main_controller.process_image(args.image_path, args.task, args.lang, args.question)
    print(final_result)

if __name__ == "__main__":
    main()
