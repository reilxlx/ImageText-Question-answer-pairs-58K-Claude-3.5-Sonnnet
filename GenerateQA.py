# -*- coding: gbk -*-
import os
import random
import shutil
import re
import json
import requests
import base64
import time
from tqdm import tqdm
from json_repair import repair_json

SOURCE_DIR = "I:\Visual Genome Dataset V1.2\data\VG_100K"
TARGET_DIR = "I:\Visual Genome Dataset V1.2\data\claude-3-5-sonnet-20240620\ProcessedImages"
JSONL_FILE = "I:\Visual Genome Dataset V1.2\data\claude-3-5-sonnet-20240620\QuestionsAnswers.jsonl"
ERROR_FILE = "I:\Visual Genome Dataset V1.2\data\claude-3-5-sonnet-20240620\ErrorLog.txt"
API_KEY = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

API_URL = "https://XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX/v1/messages"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
    "anthropic-version": "2023-06-01"
}

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_random_images(source_dir, num_images=8000):
    all_images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    return random.sample(all_images, min(num_images, len(all_images)))

def fix_json(text):
    json_objects = text.split('|+|')
    fixed_objects = []

    for obj in json_objects:
        pattern = r'("question":|"answer":)\s*(.*?)(?=,\s*"|\})'

        def replace_func(match):
            key, value = match.groups()
            if not (value.startswith('"') and value.endswith('"')):
                value = json.dumps(value, ensure_ascii=False)
            return f'{key} {value}'

        fixed_json = re.sub(pattern, replace_func, obj, flags=re.DOTALL)

        try:
            parsed = json.loads(fixed_json)
            fixed_objects.append(json.dumps(parsed, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            fixed_objects.append(fixed_json)

    return '|+|'.join(fixed_objects)

def process_image(image_path, image_name):
    prompt = """我想你充当一个识别图像内容的专家，然后根据图像内容随机提出一个或多个问题，然后给出正确回答的专家。我需要你帮助我制作一个数据集，该数据集的作用是利用图像产生众多问题和回答，我将利用该数据集训练一个多模态大模型。你至少需要提出3个问题和答案。其中的第一个问题必须是：描述图像中的信息，需要包括其中的细节和构成。你可以对该问题润色，但主题都必须与完整内容描述有关。
    回答的格式严格按照以下json要求，且仅输出json信息。如果你识别到了图片中的文字，请不要使用双引号""包围，直接描述。
    {"question": value, "answer": value}
    且每个json之间使用|+|分割"""

    image_base64 = image_to_base64(image_path)

    payload = {
        "model": "claude-3-5-sonnet-20240620",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}}
                ]
            }
        ]
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()

    content = response.json()['content'][0]['text']
    print(content)
    split_count = len(content.split('|+|'))
    qa_pairs = []
    parse_error = False

    fixed_content = fix_json(content)

    for json_str in fixed_content.split('|+|'):
        json_str = json_str.strip()
        if json_str.startswith('{') and json_str.endswith('}'):
            try:

                repaired_json_str = repair_json(json_str)
                qa_pair = json.loads(repaired_json_str)
                qa_pair = {"id": image_name, **qa_pair}
                qa_pairs.append(qa_pair)
            except json.JSONDecodeError as e:
                parse_error = True
                print(f"无法解析JSON: {json_str}")
                print(f"错误信息: {str(e)}")

    if parse_error:
        with open(ERROR_FILE, 'a', encoding='utf-8') as error_file:
            error_file.write(f"{image_name}+{content}\n")

    return qa_pairs, split_count


def move_and_update_time(source_path, target_path):
    shutil.move(source_path, target_path)
    current_time = time.time()
    os.utime(target_path, (current_time, current_time))


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)

    random_images = get_random_images(SOURCE_DIR)

    for image in tqdm(random_images, desc="处理图片"):
        source_path = os.path.join(SOURCE_DIR, image)
        target_path = os.path.join(TARGET_DIR, image)

        try:
            qa_pairs, split_count = process_image(source_path, image)

            with open(JSONL_FILE, mode='a', encoding='utf-8') as jsonl_file:
                for qa_pair in qa_pairs:
                    json.dump(qa_pair, jsonl_file, ensure_ascii=False)
                    jsonl_file.write('\n')

            if split_count != len(qa_pairs):
                print(
                    f"警告：图片 {image} 的split条数 ({split_count}) 与写入JSONL的条数 ({len(qa_pairs)}) 不一致")
                with open(ERROR_FILE, 'a', encoding='utf-8') as error_file:
                    error_file.write(
                        f"{image}: split条数 ({split_count}) 与JSONL条数 ({len(qa_pairs)}) 不一致\n")
            move_and_update_time(source_path, target_path)

        except Exception as e:
            print(f"处理图片 {image} 时出错: {str(e)}")
            with open(ERROR_FILE, 'a', encoding='utf-8') as error_file:
                error_file.write(f"{image}+{str(e)}\n")

        time.sleep(5)

if __name__ == "__main__":
    main()