from langchain.embeddings.base import Embeddings
import requests
import os
import time
from typing import List


class SiliconFlowEmbeddings(Embeddings):
    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        endpoint: str = None,
        max_batch_size: int = 64,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
    ):
        self.model = model
        self.api_key = api_key
        self.url = endpoint or "https://api.siliconflow.cn/v1/embeddings"
        self.max_batch_size = int(max_batch_size)
        self.timeout_seconds = int(timeout_seconds)
        self.max_retries = int(max_retries)
        self.backoff_base_seconds = float(backoff_base_seconds)

    def _request_embeddings(self, inputs: List[str]) -> List[List[float]]:
        """
        调用 SiliconFlow Embeddings 接口，支持重试与指数退避。
        返回与 inputs 等长的嵌入列表。
        """
        assert isinstance(inputs, list) and inputs, "inputs 不能为空列表"
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": inputs,
                    },
                    timeout=self.timeout_seconds,
                )
                resp.raise_for_status()
                data = resp.json()
                emb_list = [item["embedding"] for item in data.get("data", [])]
                if len(emb_list) != len(inputs):
                    raise ValueError(f"返回数量与输入不一致: {len(emb_list)} vs {len(inputs)}")
                return emb_list
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    sleep_s = self.backoff_base_seconds * (2 ** attempt)
                    print(f"[SiliconFlowEmbeddings] 请求失败({e}); {sleep_s}s后重试 {attempt+1}/{self.max_retries}")
                    time.sleep(sleep_s)
                else:
                    print(f"[SiliconFlowEmbeddings] 最终失败: {e}")
        raise last_err

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        results: List[List[float]] = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            batch_embs = self._request_embeddings(batch)
            results.extend(batch_embs)
        return results

    def embed_query(self, text: str) -> List[float]:
        return self._request_embeddings([text])[0]
