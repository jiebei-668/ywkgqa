from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import modelscope
import torch
import asyncio
from http import HTTPStatus
import platform
import time
import requests

from dashscope import Generation
from dashscope.aigc.generation import AioGeneration

from prompts import system_prompt


# provider可选的有[glm,baichuan,qwen, ollama],其中只有ollama是本地部署，其他都是调用api
# model_name_or_path，api调用的话就是模型名称，本地ollama调用的话就是url地址
# api_key: 调用api专属
# base_url: 调用api专属
class ChatModel:
    def __init__(self, provider: str, model_name: str, api_key: str = None, base_url: str = None, **kwargs):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        # 将额外的生成参数（如 temperature, top_p）保存下来
        self.generation_args = kwargs
        self._load_model()
        

    def _load_model(self):
        """ 根据 provider 分发加载逻辑 """
        print(f"正在加载模型: [{self.provider}] {self.model_name}...")
        
        if self.provider == "qwen":  # 阿里 Qwen
            # TODO 使用modelscope完成baichuan的api
            pass
        if self.provider == "glm":  
            # TODO 使用modelscope完成glm的api
            pass
        if self.provider == "baichuan":  
            # TODO 使用modelscope完成baichuan的api
            pass
        elif self.provider == "ollama":
            # 从 kwargs 获取 url，给一个默认值
            self.ollama_base_url = self.base_url
            self._test_ollama_connection()

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _test_ollama_connection(self):
        """测试Ollama连接"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            if response.status_code == 200:
                print(f"Ollama连接成功，可用模型: {response.json()['models']}")
            else:
                raise ConnectionError(f"Ollama连接失败，状态码: {response.status_code}")
        except Exception as e:
            raise ConnectionError(f"无法连接到Ollama服务: {e}")

    # ==========================================
    # 修改点 2: 重构 chat_ 为单轮对话入口
    # 参数 messages: 这里现在接收的是一个单纯的字符串 (Prompt)
    # ==========================================
    def chat_(self, prompt_text: str):
        """ 自定义chat接口 (单轮) """

        # 1. 构建本次对话的上下文 (System + User)
        # 这里的 messages 仅仅包含 system_prompt 和 当前用户的 prompt_text
        current_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text}
        ]

        # 2. 分发给具体的模型函数
        if self.provider == "glm":
            return self.chat_glm(current_messages)
        elif self.provider == "baichuan":
            return self.chat_baichuan(current_messages)
        elif self.provider == "qwen":
            return self.chat_qwen(current_messages)
        elif self.provider == "ollama":
            return self.chat_ollama(current_messages)
        else:
            raise ValueError("provider must be in ['glm', 'baichuan', 'qwen', 'ollama']")


    # ==========================================
    # 修改点 3: 重构 chat_ollama 接收完整列表
    # ==========================================
    def chat_ollama(self, messages_list):
        """ ollama调用chat """
        
        # 1. 准备参数
        options = {}
        options['temperature'] = self.generation_args.get('temperature', 0.1)
        for k, v in self.generation_args.items():
            if k != 'temperature': 
                options[k] = v

        # 2. 构建 Payload
        # 注意：这里直接使用 chat_ 传进来的 messages_list
        payload = {
            "model": self.model_name,
            "messages": messages_list, 
            "stream": False,
            "options": options
        }

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                assistant_response = result['message']['content']
                
                # 修改点：单轮对话通常只需要返回回答文本即可
                return assistant_response 
            else:
                print(f"Ollama API调用失败，状态码: {response.status_code}")
                return "error, no correct response"

        except Exception as e:
            print(f"Ollama调用异常: {e}")
            return "error, no correct response"

    def chat_glm(self, messages):
        # TODO 完善chat_glm
        return "", ""
    
    def chat_baichuan(self, messages):
        # TODO 完善chat_baichuan
        return "", ""
    
    def chat_qwen(self, messages):
        # TODO 完善chat_qwen
        return "", ""


def test_ollama_chat():
    """测试Ollama聊天功能"""
    print("开始测试Ollama聊天功能...")

    # 测试参数
    provider = "ollama"
    base_url = "http://localhost:11434"
    model_name = "qwen3:8b"

    try:
        # 初始化模型
        chat_model = ChatModel(provider=provider, model_name=model_name, base_url=base_url)
        print("Ollama模型初始化成功！")

        # 测试单轮对话
        print("\n=== 测试单轮对话 ===")
        # 注意：这里现在直接传字符串，而不是列表
        user_input = "你好，请介绍一下你自己"

        response = chat_model.chat_(user_input)

        print(f"用户: {user_input}")
        print(f"助手: {response}")

        print("\n=== Ollama聊天功能测试完成 ===")
        return True

    except Exception as e:
        print(f"Ollama测试失败: {e}")
        return False


if __name__ == "__main__":

    # 1 测试Ollama功能
    test_ollama_chat()

