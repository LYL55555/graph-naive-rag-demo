import logging
from dotenv import load_dotenv, find_dotenv
from langchain.chains.question_answering import load_qa_chain
#from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.callbacks.manager import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from model.get_models import get_embeddings_model
from typing import List, Tuple
import os
import pickle
from langchain_community.llms import Tongyi

_ = load_dotenv(find_dotenv())
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 初始化 Logger 对象
logger = logging.getLogger(__name__)

# 设置日志级别
logger.setLevel(logging.WARNING)
def extract_text_with_line_numbers(file_path: str) -> Tuple[str, List[int]]:
    """
    从文本文件中提取文本并记录每行文本对应的行号

    参数:
        file_path: 文本文件路径

    返回:
        text: 提取的文本内容
        line_numbers: 每行文本对应的行号列表
    """
    text = ""
    line_numbers = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                if line.strip():  # 忽略空行
                    text += line
                    line_numbers.append(line_number)
                else:
                    logger.warning(f"第 {line_number} 行为空行。")
    except UnicodeDecodeError:
        # 如果UTF-8解码失败，尝试其他编码
        try:
            with open(file_path, 'r', encoding='gbk') as file:
                for line_number, line in enumerate(file, start=1):
                    if line.strip():
                        text += line
                        line_numbers.append(line_number)
                    else:
                        logger.warning(f"第 {line_number} 行为空行。")
        except Exception as e:
            logger.error(f"无法读取文件 {file_path}: {e}")
            return "", []

    return text, line_numbers


def process_text_with_splitter(text: str, line_numbers: List[int], save_path: str = None) -> FAISS:
    """
    处理文本并创建向量存储

    参数:
        text: 提取的文本内容
        line_numbers: 每行文本对应的行号列表
        save_path: 可选，保存向量数据库的路径

    返回:
        knowledgeBase: 基于FAISS的向量存储对象
    """
    # 创建文本分割器，用于将长文本分割成小块
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )

    # 分割文本
    chunks = text_splitter.split_text(text)
    print(f"文本被分割成 {len(chunks)} 个块。")

    # 创建嵌入模型
    # embeddings = DashScopeEmbeddings(
    #     model="text-embedding-v1",
    #     dashscope_api_key=DASHSCOPE_API_KEY,
    # )
    embeddings = get_embeddings_model()

    # 创建FAISS向量存储
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    
    # 存储每个文本块对应的行号信息(行数记录有误，需调整)
    line_info = {chunk: line_numbers[i] if i < len(line_numbers) else "未知" for i, chunk in enumerate(chunks)}
    knowledgeBase.line_info = line_info

    # 如果提供了保存路径，则保存向量数据库和行号信息
    if save_path:
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)

        # 保存FAISS向量数据库
        knowledgeBase.save_local(save_path)
        print(f"向量数据库已保存到: {save_path}")

        # 保存行号信息到同一目录
        with open(os.path.join(save_path, "line_info.pkl"), "wb") as f:
            pickle.dump(line_info, f)
        print(f"行号信息已保存到: {os.path.join(save_path, 'line_info.pkl')}")

    return knowledgeBase


def load_knowledge_base(load_path: str, embeddings=None) -> FAISS:
    """
    从磁盘加载向量数据库和行号信息

    参数:
        load_path: 向量数据库的保存路径
        embeddings: 可选，嵌入模型。如果为None，将创建一个新的DashScopeEmbeddings实例

    返回:
        knowledgeBase: 加载的FAISS向量数据库对象
    """
    # 如果没有提供嵌入模型，则创建一个新的
    if embeddings is None:
        embeddings = get_embeddings_model()
    #加载FAISS向量数据库，添加allow_dangerous_deserialization=True参数以允许反序列化
    knowledgeBase = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    print(f"向量数据库已从 {load_path} 加载。")

    # 加载行号信息
    line_info_path = os.path.join(load_path, "line_info.pkl")
    if os.path.exists(line_info_path):
        with open(line_info_path, "rb") as f:
            line_info = pickle.load(f)
        knowledgeBase.line_info = line_info
        print("行号信息已加载。")
    else:
        print("警告: 未找到行号信息文件。")

    return knowledgeBase


def chat_with_text_file(file_path: str, query: str, save_path: str = None, load_existing: bool = False):
    """
    与文本文件进行对话的主函数

    参数:
        file_path: 文本文件路径
        query: 查询问题
        save_path: 可选，保存向量数据库的路径
        load_existing: 是否加载已存在的向量数据库
    """
    if load_existing and save_path and os.path.exists(save_path):
        # 加载已存在的向量数据库
        knowledgeBase = load_knowledge_base(save_path)
        print("已加载现有的向量数据库。")
    else:
        # 读取文本文件
        text, line_numbers = extract_text_with_line_numbers(file_path)
        
        if not text:
            print("错误: 无法从文件中提取文本。")
            return

        print(f"提取的文本长度: {len(text)} 个字符。")

        # 处理文本并创建知识库，同时保存到磁盘
        knowledgeBase = process_text_with_splitter(text, line_numbers, save_path=save_path)

    # 创建语言模型
    llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=DASHSCOPE_API_KEY)

    if query:
        # 执行相似度搜索，找到与查询相关的文档
        docs = knowledgeBase.similarity_search(query, k=10)

        # 加载问答链
        chain = load_qa_chain(llm, chain_type="stuff")

        # 准备输入数据
        input_data = {"input_documents": docs, "question": query}

        # 使用回调函数跟踪API调用成本
        with get_openai_callback() as cost:
            # 执行问答链
            response = chain.invoke(input=input_data)
            print(f"查询已处理。成本: {cost}")
            print("回答:")
            print(response["output_text"])
            print("\n来源:")

        # 记录唯一的行号
        unique_lines = set()

        # 显示每个文档块的来源行号
        for doc in docs:
            text_content = getattr(doc, "page_content", "")
            source_line = knowledgeBase.line_info.get(
                text_content.strip(), "未知"
            )

            if source_line not in unique_lines:
                unique_lines.add(source_line)
                print(f"文本块行号: {source_line}")


if __name__ == "__main__":
    # # 示例用法
    text_file_path = "./files/equipment_data_美国_合并输出_translated.txt"  # 替换为您的文本文件路径
    save_dir = "./txt_vector_db"
    #
    #
    # # 与文本文件对话
    # chat_with_text_file(
    #     file_path=text_file_path,
    #     query=query,
    #     save_path=save_dir,
    #     load_existing=False  # 设置为True可以加载已存在的向量数据库
    # )

    # 示例：如何加载已保存的向量数据库进行新的查询

    # 加载已保存的向量数据库
    # loaded_knowledgeBase = load_knowledge_base(save_dir)

    # 新的查询
    new_query = "“九头蛇-70”航空火箭系统的装备类型是？"
    chat_with_text_file(
        file_path=text_file_path,
        query=new_query,
        save_path=save_dir,
        load_existing=True
    )

    # import faiss
    # faiss_index_path = './txt_vector_db/index.faiss'
    # mapping_path = './txt_vector_db/line_info.pkl'
    #
    # # 1. 加载FAISS索引
    # index = faiss.read_index(faiss_index_path)
    # print("索引类型:", type(index))
    # print("向量维度:", index.d)
    # print("索引中向量数量:", index.ntotal)
    #
    # 2. 加载chunk_id与索引号的映射
    # with open(mapping_path, 'rb') as f:
    #     mapping = pickle.load(f)
    # print(mapping)