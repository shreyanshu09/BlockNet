#!/usr/bin/env python
# coding: utf-8

## Variables

# OpenAI API Key
api_key = 'api_key'

# Global Model variables
global_model_path = 'global_model/block_diagram_global_information'
empty_folder = 'global_model/block_diagram_global_information/dataset/c2t_data/'    # create an empty folder
global_limit_token = 500

# OCR Variable
pororo_path = 'ocr_pororo'

# Local Model Variables variables
local_model_path = 'local_model/block_diagram_symbol_detection/symbol_detection'
object_detection_output_path = 'local_model/block_diagram_symbol_detection/symbol_detection/runs/detect/exp/labels'
yolo_weights_path = 'local_model/block_diagram_symbol_detection/symbol_detection/runs/train/best_all/weights/best.pt'
yolo_yaml_file = 'local_model/block_diagram_symbol_detection/symbol_detection/data/mydata.yaml'


## Global Infromation Extractor

import os
from PIL import Image
import torch
from donut import DonutModel

# Load the pre-trained model
global_model = DonutModel.from_pretrained(global_model_path) 

# Move the model to GPU if available
if torch.cuda.is_available():
    global_model.half()
    device = torch.device("cuda:0")
    global_model.to(device)
    
# Function to process a single image
def global_model_process(image):

    try:
        # Load and process the image
        image1 = Image.fromarray(image)
        task_name = os.path.basename(empty_folder)  
        result = global_model.inference(image=image1, prompt=f"<s_{task_name}>")["predictions"][0]

        # Extract the relevant information from the result
        if 'c2t' in result:
            text_result = result['c2t']
        else:
            text_result = result['text_sequence']

        # Limit the result to 500 tokens
        limited_result = ' '.join(text_result.split()[:global_limit_token])

        return limited_result

    except Exception as e:
        # Return an empty string in case of an error
        return ""


## Local Information Extractor

### Object Detection

import argparse
import os
from pathlib import Path
import torch

import sys
sys.path.append(local_model_path)

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2, increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors, save_one_box, save_block_box
from utils.torch_utils import select_device, smart_inference_mode

def load_model(weights, device, dnn, data, fp16):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=fp16)
    return model

def run_single_image_inference(model, img_path, stride, names, pt, conf_thres=0.35, iou_thres=0.7, max_det=100, augment=True, visualize=False, line_thickness=1, hide_labels=False, hide_conf=False, save_conf=False, save_crop=False, save_block=True, imgsz=(640, 640), vid_stride=1, bs=1, classes=None, agnostic_nms=False, save_txt=True, save_img=True):
    dataset = LoadImages(img_path, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # Load image from file
    imgsz = check_img_size(imgsz, s=stride) 

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        sorted_data_list = []

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop or save_block else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                data_for_image=[]
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    data_for_image.append((int(cls), xywh))
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Sort the data based on the top-left coordinates (Y first, then X)
            sorted_data_for_image = sorted(data_for_image, key=lambda x: (x[1][1], x[1][0]))
            sorted_data_list.extend(sorted_data_for_image)
    
    # Return the combined sorted data as a tuple
    return tuple(sorted_data_list)

yolo_model = load_model(yolo_weights_path, device='cpu', dnn=False, data=yolo_yaml_file, fp16=False)
stride, names, pt = yolo_model.stride, yolo_model.names, yolo_model.pt


### BlockSplit: Break image into smaller units

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

def read_list(annotation_list, image):
    image_height, image_width = image.shape[:2]
    edge = []
    node = []
    for annotation in annotation_list:
        category, bbox_norm = annotation

        x_norm, y_norm, w_norm, h_norm = bbox_norm

        x = x_norm * image_width
        y = y_norm * image_height
        w = w_norm * image_width
        h = h_norm * image_height

        if category == 0:
            if w < h:
                edge1 = (x, y - h/2)
                edge2 = (x, y + h/2)
                edge.append([(x, y, w, h), edge1, edge2])
            elif w >= h:
                edge1 = (x - w/2, y)
                edge2 = (x + w/2, y)
                edge.append([(x, y, w, h), edge1, edge2])
        elif category in [1, 2, 3, 5]:
            t = (x, y + h/2)
            b = (x, y - h/2)
            l = (x - w/2, y)
            r = (x + w/2, y)
            node.append([(x, y, w, h), t, b, l, r])

    return edge, node

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def calculate_relative_position(edge_point, node_point):
    # Calculate the relative position of the node with respect to the edge point
    dx = node_point[0] - edge_point[0]
    dy = node_point[1] - edge_point[1]

    # Return the relative position as a tuple (dx, dy)
    return dx, dy

def find_closest_node(edge, node):
    results = []
    for edge_box in edge:
        edge1, edge2 = edge_box[1], edge_box[2]
        min_distance1 = float('inf')
        min_distance2 = float('inf')
        closest_node1 = None
        closest_node2 = None

        for node_box in node:
            for i in range(4):
                node_point = node_box[1 + i]

                distance1 = calculate_distance(edge1, node_point)
                distance2 = calculate_distance(edge2, node_point)

                if distance1 < min_distance1:
                    min_distance1 = distance1
                    closest_node1 = node_box

                if distance2 < min_distance2:
                    min_distance2 = distance2
                    closest_node2 = node_box

        # Calculate relative positions of closest nodes with respect to edges
        rel_pos1 = calculate_relative_position(edge1, closest_node1[0])
        rel_pos2 = calculate_relative_position(edge1, closest_node2[0])

        # Choose the closest node based on relative positions
        if rel_pos1[0] < 0 or rel_pos1[1] < 0:
            # If closest_node1 is to the left or above edge1, prefer it
            results.append(('edge_box', edge_box, 'closest_node1', closest_node1, 'closest_node2', closest_node2))
        else:
            # Otherwise, prefer closest_node2
            results.append(('edge_box', edge_box, 'closest_node1', closest_node2, 'closest_node2', closest_node1))

    return results


# Extract Triplets

import os
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import numpy as np

import sys
sys.path.append(pororo_path)

from main import PororoOcr

ocr = PororoOcr()
ocr.get_available_langs()
ocr.get_available_models()

def pororo_ocr(img_path):
    ocr.run_ocr(img_path, debug=False) 
    res = ocr.get_ocr_result()
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
        word_coordinates.append((word, (x_min, y_min, width, height)))   ## (word, (x,y,w,h))

    return word_coordinates

def process_edge_box(image, edge_box):
    # Extract the coordinates and dimensions from the edge box
    x_mid, y_mid, w, h = map(int, edge_box)
    x1, y1 = x_mid - w // 2, y_mid - h // 2
    x2, y2 = x1 + w, y1 + h

    # Crop the image to the specified region
    roi_edge = image[y1:y2, x1:x2]

    # Convert to grayscale
    gray_roi = cv2.cvtColor(roi_edge, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to the cropped region
    _, thresholded_edge = cv2.threshold(gray_roi, 128, 255, cv2.THRESH_BINARY)

    # Invert the colors (make the background black and the object white)
    inverted_edge = cv2.bitwise_not(thresholded_edge)

    return inverted_edge

def find_head_tail(thresholded_edge, closest_node1_text, closest_node2_text):
    h, w = thresholded_edge.shape

    if h > w:
        half1 = thresholded_edge[:h // 2, :]
        half2 = thresholded_edge[h // 2:, :]
    else:
        half1 = thresholded_edge[:, :w // 2]
        half2 = thresholded_edge[:, w // 2:]

    # Ensure single-channel
    half1 = cv2.cvtColor(half1, cv2.COLOR_BGR2GRAY) if len(half1.shape) == 3 else half1
    half2 = cv2.cvtColor(half2, cv2.COLOR_BGR2GRAY) if len(half2.shape) == 3 else half2

    white_pixels_half1 = cv2.countNonZero(half1)
    white_pixels_half2 = cv2.countNonZero(half2)

    if white_pixels_half1 > white_pixels_half2:
        # Perform swap 
        return closest_node2_text, closest_node1_text
    else:
        # Return original head-tail if not swapping
        return closest_node1_text, closest_node2_text

def calculate_distance2(box1, box2):
    x1, y1, w1, h1 = box1
    x2_mid, y2_mid, w2, h2 = box2
    
    # Calculate the coordinates of the center of box2
    x2 = x2_mid - w2 / 2
    y2 = y2_mid - h2 / 2

    center1 = np.array([x1 + w1 / 2, y1 + h1 / 2])
    center2 = np.array([x2 + w2 / 2, y2 + h2 / 2])

    distance = np.linalg.norm(center1 - center2)
    return distance

# Function to extract all text from specific coordinates using pororoocr results
def extract_text_from_coordinates(coordinates, ocr_result):
    x_mid, y_mid, w, h = coordinates

    # Calculate the top-left and bottom-right corners of the bounding box
    x = x_mid - w // 2
    y = y_mid - h // 2

    # Collect all words that lie inside the specified coordinates
    matching_words = [word for word, (word_x, word_y, word_w, word_h) in ocr_result
                      if x <= word_x <= x + w and y <= word_y <= y + h]

    # Combine the matching words into a single string separated by spaces
    combined_text = ' '.join(matching_words)

    return combined_text, matching_words

def process_json_file(image, json_result):

    try:
        data = json_result

        ocr_result = pororo_ocr(image)

        # Initialize empty list for triplets
        triplets = []

        # Iterate through each result in the list
        for result in data['results']:
            # Find the index of 'edge_box', 'closest_node1', and 'closest_node2'
            edge_box_index = result.index('edge_box')
            closest_node1_index = result.index('closest_node1')
            closest_node2_index = result.index('closest_node2')

            # Extract the coordinates of edge_box, closest_node1, and closest_node2
            edge_box = result[edge_box_index + 1][0]
            closest_node1 = result[closest_node1_index + 1][0]
            closest_node2 = result[closest_node2_index + 1][0]

            # Append the triplet to the list
            triplets.append({'edge_box': edge_box, 'closest_node1': closest_node1, 'closest_node2': closest_node2})

        formatted_triplets = []
        used_words = set() 

        # find head, tail
        for triplet in triplets:
            closest_node1_coords = triplet['closest_node1']
            closest_node2_coords = triplet['closest_node2']

            edge_box = triplet['edge_box']

            # Extract text using pororoocr results
            closest_node1_text, closest_node1_words = extract_text_from_coordinates(closest_node1_coords, ocr_result)
            closest_node2_text, closest_node2_words = extract_text_from_coordinates(closest_node2_coords, ocr_result)

            # Add used words to the set
            used_words.update(closest_node1_words)
            used_words.update(closest_node2_words)

            # Process the edge box using the separate function
            thresholded_edge = process_edge_box(image, edge_box)
            head, tail = find_head_tail(thresholded_edge, closest_node1_text, closest_node2_text)

            # Add used words to the set
            used_words.update(head.split())
            used_words.update(tail.split())

            # Check if any Korean characters are present using regular expression
            if re.search('[\u3131-\u3163\uac00-\ud7a3]+', head + tail):
                relation = "와 연계된"
            else:
                relation = "connected with"  # Fallback to English if no Korean characters are found

            # Format the output
            output_triplet = f"<H> {head} <R> {relation} <T> {tail}"

            # Append the formatted triplet to the list
            formatted_triplets.append(output_triplet)

        # Find relations
        relation_word_info = [word_info for word_info in ocr_result if word_info[0] not in used_words]
        nearest_edge_box_indexes = []

        for relation_info in relation_word_info:
            word, word_box = relation_info
            distances = []

            for triplet in triplets:
                edge_box = triplet['edge_box']
                distance = calculate_distance2(word_box, edge_box)
                distances.append(distance)
                
            nearest_edge_box_index = np.argmin(distances)
            nearest_edge_box_indexes.append(nearest_edge_box_index)

        # Iterate through the nearest_edge_box_indexes and relation_word_info simultaneously
        for nearest_edge_box_index, (relation_word, _) in zip(nearest_edge_box_indexes, relation_word_info):
            # Check the language of the relation word
            if re.search('[\u3131-\u3163\uac00-\ud7a3]+', head + tail):
                # English text, update using 'connected with'
                formatted_triplets[nearest_edge_box_index] = formatted_triplets[nearest_edge_box_index].replace('<R> 와 연계된', f'<R> {relation_word}')
            else:
                # Korean text, update using '와 연계된'
                formatted_triplets[nearest_edge_box_index] = formatted_triplets[nearest_edge_box_index].replace('<R> connected with', f'<R> {relation_word}')
                
        return formatted_triplets, ocr_result
    
    except Exception as e:
        ocr_output = pororo_ocr(image)
        formatted_triplets = ''
        return formatted_triplets, ocr_output


## Integration (GPT4v) Description

import base64
import requests

def gpt4(base64_image, local_model_output, global_model_output, ocr, lang, todo):

    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }

    payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"Your task is to generate {todo} of the given block diagram image with the help of Reference summary, Reference triplets and the OCR outputs (word, [x,y,w,h]) only in {lang} Language without mentioning about these helps in the Output. \nReference summary: {global_model_output} \nReference triplets: {local_model_output} \nOCR Output: {ocr}"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "temperature": 0, 
      "max_tokens": 2000,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    response_json = response.json()

    # Extract 'content' from the JSON
    content = response_json['choices'][0]['message']['content']

    return content


## Integration (GPT4v) QA

import base64
import requests

def gpt4qa(base64_image, local_model_output, global_model_output, ocr, lang, question, todo):

    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }

    payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"Your task is to answer the given Question {todo} based on the given block diagram image with the help of Reference summary, Reference triplets and the OCR outputs (word, [x,y,w,h]) only in {lang} Language without mentioning about these helps in the Output. \nQuestion: {question} \nReference summary: {global_model_output} \nReference triplets: {local_model_output} \nOCR Output: {ocr}"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "temperature": 0, 
      "max_tokens": 2000,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    response_json = response.json()

    # Extract 'content' from the JSON
    content = response_json['choices'][0]['message']['content']

    return content


## Main

import subprocess
import numpy as np
import base64
import io
import tempfile
import cv2
from PIL import Image

def process_image(input_image, task="Short Description", lang="Korean", question=None):

    img = np.array(input_image)
    # OCR
    ocr_output = pororo_ocr(img)

    ## Global Information Extractor
    global_model_output = global_model_process(input_image)

    ## Local Information Extractor
    # Object Detection
    # Save Gradio input image to a temporary file
    temp_image_path = tempfile.mktemp(suffix=".jpg")

    # Convert the image to RGB mode before saving
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(input_image_rgb)
    pil_image.save(temp_image_path, format="JPEG")

    # sort output generated from Object Detection
    txt_data_labels = run_single_image_inference(yolo_model, temp_image_path, stride, names, pt)

    # Cleanup: Remove the temporary image file
    os.remove(temp_image_path)

    # Extract triplets
    img = np.array(input_image)
    edge, node = read_list(txt_data_labels, img)
    crop_image_result = find_closest_node(edge, node)
    json_result = {
        'results': crop_image_result
    }

    # Find head, relation, tail
    local_model_output, ocr_output = process_json_file(img, json_result)

    # Assuming 'input_image' is your NumPy array representing the image
    img_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Convert the NumPy array to a PIL image
    pil_image2 = Image.fromarray(np.uint8(img_rgb))

    # Save the PIL image to a BytesIO object in JPEG format
    image_bytes = io.BytesIO()
    pil_image2.save(image_bytes, format='JPEG')

    # Get the bytes from the BytesIO object
    image_data = image_bytes.getvalue()

    # Encode bytes to base64
    encode_image = base64.b64encode(image_data).decode('utf-8')

    if task == "Short QA":
        final_result = gpt4qa(encode_image, local_model_output, global_model_output, ocr_output, lang, question, todo='very shortly')
    elif task == "Long QA":
        final_result = gpt4qa(encode_image, local_model_output, global_model_output, ocr_output, lang, question, todo='in detail')
    elif task == "Short Description":
        final_result = gpt4(encode_image, local_model_output, global_model_output, ocr_output, lang, todo='a very short Description in one paragraph only')
    else:
        final_result = gpt4(encode_image, local_model_output, global_model_output, ocr_output, lang, todo='the Description in detail')
        

    return final_result


## Gradio app

import gradio as gr

# Provide sample images as examples
sample_images = [
    "test_sample/155502.png",
    "test_sample/155958.png",
    "test_sample/160132.png",
    "test_sample/kor_real_world_292.jpg",
    "test_sample/kor_flowchart_6458.jpg",
    "test_sample/kor_graphlr_30.jpg",
    "test_sample/Connect (19).png",
    "test_sample/eng_flowchart_1369.jpg",
    "test_sample/eng_flowchart_2726.jpg",
    # Add more sample image paths as needed
]

# Create a Gradio interface with custom image display settings and two dropdowns for task and language selection
iface = gr.Interface(
    fn=process_image,
    inputs=[
        "image",
        gr.Dropdown(["Short Description", "Long Description", "Short QA", "Long QA"], label="Select Task"),
        gr.Dropdown(["Korean", "English"], label="Select Language"),
        gr.Textbox(label="Enter Question (QA)", placeholder="Type your question here only for QA Task", visible=True)
    ],
    outputs="text",
    examples=[[sample_images[0]], [sample_images[1]], [sample_images[2]], [sample_images[3]], [sample_images[4]], [sample_images[5]], [sample_images[6]], [sample_images[7]], [sample_images[8]]],
    examples_per_page=len(sample_images),
    title="Block Diagram Assistant",
    description="Block Diagram Image"
)

# Launch the Gradio interface
iface.launch(share=True)

