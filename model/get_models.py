from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain_community.embeddings import DashScopeEmbeddings

import os
import torch
from pathlib import Path
from dotenv import load_dotenv

# 设置tiktoken缓存避免网络问题
def setup_cache():
    cache_dir = Path.home() / "cache" / "tiktoken"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)

setup_cache()

load_dotenv()

from model.siliconflow_embeddings import SiliconFlowEmbeddings

def get_embeddings_model():
    model_name = os.getenv("SILICONFLOW_MODEL")
    api_key = os.getenv("SILICONFLOW_API_KEY")
    model = SiliconFlowEmbeddings(
        model=model_name,
        api_key=api_key,
    )
    # 屏蔽完整密钥，仅显示后四位
    masked = (api_key[-4:] if api_key else "")
    print(f"当前使用 SiliconFlow Embedding 模型: {model_name}，API Key 尾号: {masked}")
    return model

# def get_embeddings_model():
#     model = OpenAIEmbeddings(
#         model=os.getenv('OPENAI_EMBEDDINGS_MODEL'),
#         api_key=os.getenv('OPENAI_API_KEY'),
#         base_url=os.getenv('OPENAI_BASE_URL'),
#     )
#     return model

# def get_embeddings_model():
#     model = DashScopeEmbeddings(
#         model=os.getenv("OPENAI_EMBEDDINGS_MODEL"),
#         dashscope_api_key=os.getenv("OPENAI_API_KEY")
# )
#     print("[INFO]使用阿里云 Dashscope embedding 成功")
#     return model

# from langchain_community.embeddings import HuggingFaceEmbeddings

# def get_embeddings_model():
#     model = HuggingFaceEmbeddings(
#         model_name="BAAI/bge-m3",
#         model_kwargs={"device": "cuda:0" if torch.cuda.is_available() else "cpu"},
#         encode_kwargs={"normalize_embeddings": True} 
#     )
#     return model

def get_llm_model():
    model_name = os.getenv('OPENAI_LLM_MODEL')
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_BASE_URL')
    # 处理数值类型
    temp_env = os.getenv('TEMPERATURE')
    max_tokens_env = os.getenv('MAX_TOKENS')
    try:
        temperature = float(temp_env) if temp_env is not None else 0.2
    except Exception:
        temperature = 0.2
    try:
        max_tokens = int(max_tokens_env) if max_tokens_env is not None else 2048
    except Exception:
        max_tokens = 2048

    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
    )
    print(f"当前使用 Chat 模型: {model_name}，Base URL: {base_url}")
    return model

def get_stream_llm_model():
    callback_handler = AsyncIteratorCallbackHandler()
    # 将回调handler放进AsyncCallbackManager中
    manager = AsyncCallbackManager(handlers=[callback_handler])

    model = ChatOpenAI(
        model=os.getenv('OPENAI_LLM_MODEL'),
        temperature=os.getenv('TEMPERATURE'),
        max_tokens=os.getenv('MAX_TOKENS'),
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_URL'),
        streaming=True,
        callbacks=manager,
    )
    return model

def count_tokens(text):
    """简单通用的token计数"""
    if not text:
        return 0
    
    model_name = os.getenv('OPENAI_LLM_MODEL', '').lower()
    
    # 如果是deepseek，使用transformers
    if 'deepseek' in model_name:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
            return len(tokenizer.encode(text))
        except:
            pass
    
    # 如果是gpt，使用tiktoken
    if 'gpt' in model_name:
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except:
            pass
    
    # 备用方案：简单计算
    chinese = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english = len(text) - chinese
    return chinese + english // 4

if __name__ == '__main__':
    # 测试llm
    llm = get_llm_model()
    print(llm.invoke("你好"))

    # 由于langchain版本问题，这个目前测试会报错
    # llm_stream = get_stream_llm_model()
    # print(llm_stream.invoke("你好"))

    # 测试embedding
    test_text = "你好，这是一个测试。"
    embeddings = get_embeddings_model()
    print(embeddings.embed_query(test_text))

    # 测试计数
    test_text = "Hello 你好世界"
    tokens = count_tokens(test_text)
    print(f"Token计数: '{test_text}' = {tokens} tokens")
