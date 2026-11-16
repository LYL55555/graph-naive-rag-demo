import re
import os
import time
from typing import List, Tuple
from model.get_models import get_llm_model
from config.settings import FILES_DIR

class TextTranslator:
    """
    智能文本翻译器，用于处理中英文混合文档
    只翻译英文句子，保持中文内容不变，并保护专有名词
    """
    
    def __init__(self):
        self.llm = get_llm_model()
        self.token_calls = 0  # 统计token调用次数
        self.total_translated_chars = 0  # 统计翻译字符数
        
    def detect_language(self, text: str) -> str:
        """
        检测文本主要语言
        
        Args:
            text: 输入文本
            
        Returns:
            'chinese', 'english', 'mixed'
        """
        # 移除标点符号和数字进行分析
        clean_text = re.sub(r'[^\w\s]', '', text)
        clean_text = re.sub(r'\d+', '', clean_text)
        
        if not clean_text.strip():
            return 'mixed'
            
        # 统计中英文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', clean_text))
        english_chars = len(re.findall(r'[a-zA-Z]', clean_text))
        
        total_chars = chinese_chars + english_chars
        if total_chars == 0:
            return 'mixed'
            
        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars
        
        # 判断语言类型
        if chinese_ratio > 0.7:
            return 'chinese'
        elif english_ratio > 0.7:
            return 'english'
        else:
            return 'mixed'
    
    def extract_proper_nouns(self, text: str) -> List[str]:
        """
        提取专有名词（大写字母开头的词组、缩写等）
        
        Args:
            text: 输入文本
            
        Returns:
            专有名词列表
        """
        # 匹配专有名词的模式
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # 首字母大写的词组
            r'\b[A-Z]{2,}\b',  # 全大写缩写
            r'\b[A-Z][a-z]*-\d+\b',  # 型号名称如 ABC-123
            r'\b\w*[A-Z]\w*\d+\w*\b',  # 包含大写字母和数字的标识符
        ]
        
        proper_nouns = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            proper_nouns.extend(matches)
        
        return list(set(proper_nouns))
    
    def create_translation_prompt(self, text: str, proper_nouns: List[str]) -> str:
        """
        创建翻译提示词
        
        Args:
            text: 要翻译的文本
            proper_nouns: 需要保护的专有名词
            
        Returns:
            翻译提示词
        """
        protected_nouns = ", ".join(proper_nouns) if proper_nouns else "无"
        
        prompt = f"""请将以下英文文本翻译成中文，要求：

1. 翻译要准确、流畅、符合中文表达习惯
2. 保持原文的时间、数字、日期格式
3. 以下专有名词请保持原文不变：{protected_nouns}
4. 地名、人名、机构名等专有名词可以翻译，但要保持准确性
5. 只输出翻译结果，不要添加任何解释

原文：
{text}

翻译："""
        
        return prompt
    
    def translate_text(self, text: str) -> str:
        """
        翻译文本 - 只有这个函数会调用token
        
        Args:
            text: 要翻译的文本
            
        Returns:
            翻译结果
        """
        # 🔥 TOKEN调用标记 - 只有这里会消耗token
        self.token_calls += 1
        self.total_translated_chars += len(text)
        
        print(f"🔥 [TOKEN调用 {self.token_calls}] 翻译英文内容 ({len(text)} 字符): {text[:100]}...")
        
        # 提取专有名词
        proper_nouns = self.extract_proper_nouns(text)
        
        # 创建提示词
        prompt = self.create_translation_prompt(text, proper_nouns)
        
        try:
            start_time = time.time()
            # 调用大模型翻译
            response = self.llm.invoke(prompt)
            end_time = time.time()
            
            result = response.content.strip()
            print(f"✅ [TOKEN调用 {self.token_calls}] 翻译完成 ({end_time - start_time:.2f}s): {result[:100]}...")
            
            return result
        except Exception as e:
            print(f"❌ [TOKEN调用 {self.token_calls}] 翻译失败: {e}")
            return text  # 翻译失败时返回原文
    
    def process_paragraph(self, paragraph: str) -> str:
        """
        处理单个段落
        
        Args:
            paragraph: 段落文本
            
        Returns:
            处理后的段落
        """
        if not paragraph.strip():
            return paragraph
            
        # 检测语言
        language = self.detect_language(paragraph)
        
        if language == 'chinese':
            # 🟢 纯中文，不翻译，不消耗token
            print(f"🟢 [跳过] 纯中文内容: {paragraph[:50]}...")
            return paragraph
        elif language == 'english':
            # 🔥 纯英文，需要翻译，消耗token
            return self.translate_text(paragraph)
        else:
            # 🟡 混合语言，进一步处理
            print(f"🟡 [混合] 混合语言内容: {paragraph[:50]}...")
            return self.process_mixed_paragraph(paragraph)
    
    def process_mixed_paragraph(self, paragraph: str) -> str:
        """
        处理混合语言段落
        
        Args:
            paragraph: 混合语言段落
            
        Returns:
            处理后的段落
        """
        # 按句子分割
        sentences = re.split(r'[.!?。！？]', paragraph)
        processed_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 检测句子语言
            lang = self.detect_language(sentence)
            
            if lang == 'english':
                # 🔥 英文句子，翻译，消耗token
                translated = self.translate_text(sentence)
                processed_sentences.append(translated)
            else:
                # 🟢 中文或混合句子，保持不变，不消耗token
                print(f"🟢 [保持] 中文句子: {sentence[:30]}...")
                processed_sentences.append(sentence)
        
        # 重新组合句子
        result = '。'.join(processed_sentences)
        if result and not result.endswith('。'):
            result += '。'
            
        return result
    
    def translate_file(self, input_file: str, output_file: str) -> None:
        """
        翻译文件
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        try:
            # 读取文件
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"📁 开始翻译文件: {input_file}")
            print(f"📊 原文件大小: {len(content)} 字符")
            print(f"{'='*60}")
            
            # 重置统计
            self.token_calls = 0
            self.total_translated_chars = 0
            
            # 按段落处理
            paragraphs = content.split('\n')
            translated_paragraphs = []
            
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    print(f"\n📝 处理段落 {i+1}/{len(paragraphs)}")
                    translated = self.process_paragraph(paragraph)
                    translated_paragraphs.append(translated)
                else:
                    translated_paragraphs.append(paragraph)
            
            # 合并结果
            result = '\n'.join(translated_paragraphs)
            
            # 添加翻译备注
            translation_note = f"""

---
## 翻译备注
- 原文件: {os.path.basename(input_file)}
- 翻译完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
- Token调用次数: {self.token_calls}
- 翻译字符数: {self.total_translated_chars}
- 说明: 此文档由AI自动翻译，仅翻译英文部分，中文内容保持不变
"""
            
            result_with_note = result + translation_note
            
            # 写入输出文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result_with_note)
            
            print(f"\n{'='*60}")
            print(f"✅ 翻译完成！")
            print(f"📁 输出文件: {output_file}")
            print(f"📊 翻译后大小: {len(result_with_note)} 字符")
            print(f"🔥 总Token调用次数: {self.token_calls}")
            print(f"📝 总翻译字符数: {self.total_translated_chars}")
            
        except Exception as e:
            print(f"❌ 翻译文件时出错: {e}")
    
    def print_stats(self):
        """打印统计信息"""
        print(f"\n📊 翻译统计:")
        print(f"🔥 Token调用次数: {self.token_calls}")
        print(f"📝 总翻译字符数: {self.total_translated_chars}")

def main():
    """主函数"""
    import sys
    
    translator = TextTranslator()
    
    # 处理命令行参数或用户输入
    if len(sys.argv) > 1:
        # 从命令行参数获取文件名
        filename = sys.argv[1]
    else:
        # 显示可用的txt文件
        txt_dir = os.path.join(FILES_DIR, 'txt文件')
        if os.path.exists(txt_dir):
            txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt') and not f.endswith('_translated.txt')]
            if txt_files:
                print("📂 可用的txt文件:")
                for i, file in enumerate(txt_files, 1):
                    print(f"   {i}. {file}")
                print()
        
        # 提示用户输入文件名
        print("📝 请输入要翻译的文件名（不包含路径）:")
        print("💡 例如: 历史上的今天.txt")
        filename = input("文件名: ").strip()
    
    if not filename:
        print("❌ 错误：未指定文件名")
        return
    
    # 构建输入文件路径
    input_file = os.path.join(FILES_DIR, 'txt文件', filename)
    
    # 生成输出文件名（在原文件名基础上添加_translated）
    name_part, ext_part = os.path.splitext(filename)
    output_filename = f"{name_part}_translated{ext_part}"
    output_file = os.path.join(FILES_DIR, 'txt文件', output_filename)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误：输入文件不存在: {input_file}")
        print(f"💡 请确保文件位于: {os.path.join(FILES_DIR, 'txt文件')}")
        return
    
    # 执行翻译
    translator.translate_file(input_file, output_file)
    
    # 打印最终统计
    translator.print_stats()

if __name__ == "__main__":
    """
    使用方法:
    1. 交互式运行: python translate_complete.py
    2. 命令行参数: python translate_complete.py 文件名.txt
    
    示例:
    python translate_complete.py 历史上的今天.txt
    python translate_complete.py
    """
    main() 