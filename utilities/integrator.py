import base64
import requests

class IntegrationGPT4:
    def __init__(self, api_key):
        self.api_key = api_key

    def generate_answer(self, base64_image, local_model_output, global_model_output, ocr, lang, todo):
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
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
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            "temperature": 0,
            "max_tokens": 2000,
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_json = response.json()
        content = response_json['choices'][0]['message']['content']
        return content

    def answer_question(self, base64_image, local_model_output, global_model_output, ocr, lang, question, todo):
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
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
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            "temperature": 0,
            "max_tokens": 2000,
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_json = response.json()
        content = response_json['choices'][0]['message']['content']
        return content
