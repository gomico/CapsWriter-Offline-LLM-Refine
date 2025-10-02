import asyncio
import json
import httpx
from typing import Optional
from config import ClientConfig as Config
from util.client_cosmic import console


class LLMProcessor:
    def __init__(self):
        self.api_base = Config.llm_api_base
        self.api_key = Config.llm_api_key
        self.model = Config.llm_model
        self.timeout = Config.llm_timeout
        self.enable = Config.llm_enable
        self.proxy = Config.llm_proxy

    async def optimize_text(self, text: str) -> str:
        """
        使用LLM优化转录文本，修正错别字和标点符号
        """
        if not self.enable:
            return text

        if not text.strip():
            return text

        # 构建提示词
        prompt = self._build_prompt(text)

        # 构建API请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": Config.llm_system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1024
        }

        try:
            client_kwargs = {"timeout": self.timeout}
            if self.proxy:
                client_kwargs["proxy"] = self.proxy
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=data
                )
                response.raise_for_status()

                result = response.json()
                optimized_text = result["choices"][0]["message"]["content"].strip()

                console.print(f"    LLM优化: {text} -> {optimized_text}")
                return optimized_text

        except httpx.TimeoutException:
            console.print("[yellow]LLM优化超时，使用原始文本")
            return text
        except httpx.RequestError as e:
            console.print(f"[yellow]LLM请求错误: {e.request.url} - {e.__class__.__name__}: {e}，使用原始文本")
            return text
        except Exception as e:
            console.print(f"[yellow]LLM优化失败: {e.__class__.__name__}: {e}，使用原始文本")
            return text

    def _build_prompt(self, text: str) -> str:
        """
        构建发送给LLM的提示词
        """
        prompt = Config.llm_user_prompt_template.format(input_text=text)
        return prompt


# 全局LLM处理器实例
llm_processor = LLMProcessor()