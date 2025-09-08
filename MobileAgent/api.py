import os
import base64
import requests


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def inference_chat(chat, API_TOKEN):
    api_url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }

    data = {
        "model": 'gpt-4-vision-preview',
        "messages": [],
        "max_tokens": 2048,
    }

    for role, content in chat:
        data["messages"].append({"role": role, "content": content})

    while 1:
        try:
            res = requests.post(api_url, headers=headers, json=data)
            res = res.json()['choices'][0]['message']['content']
        except:
            print("Network Error:")
            print(res)
        else:
            break

    return res


def inference_chat_ollama(chat, api=None):
    """
    Ollama 版 inference_chat，接口保持和原始 OpenAI 版本一致：
    - 输入: chat ([(role, content), ...])，API_TOKEN 参数忽略（为了接口兼容）
    - 输出: 字符串 response
    """

    api_url = "http://localhost:11434/api/chat"
    model = "0000/ui-tars-1.5-7b-q8_0:7b"

    headers = {"Content-Type": "application/json"}
    messages = []

    # chat: [(role, content), ...]
    for role, content in chat:
        text_parts = []
        image_list = []

        # content 在原版里可能是字符串，也可能是复杂结构
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):  # vision 模型会传 [{"type": "text"/"image_url", ...}]
            for item in content:
                if item["type"] == "text":
                    text_parts.append(item["text"])
                elif item["type"] == "image_url":
                    url = item["image_url"]["url"]
                    if os.path.exists(url):
                        image_list.append(url)  # 本地文件路径
                    elif url.startswith("data:image"):
                        image_list.append(url.split(",")[1])  # base64 部分
                    else:
                        raise ValueError(f"Ollama only supports local path or base64, got: {url}")
                elif item["type"] == "image":  # PIL.Image
                    temp_path = "./temp_image.png"
                    item["image"].save(temp_path)
                    image_list.append(temp_path)

        msg = {
            "role": role,
            "content": "\n".join(text_parts) if text_parts else " "
        }
        if image_list:
            msg["images"] = image_list
        messages.append(msg)

    data = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 100
        }
    }

    while True:
        try:
            res = requests.post(api_url, headers=headers, json=data)
            res.raise_for_status()
            res_json = res.json()
            response = res_json["message"]["content"]
        except Exception as e:
            print("Network Error:", str(e))
            continue
        else:
            break

    return response
