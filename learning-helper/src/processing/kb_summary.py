"""
知识库总结管理
"""
import os
from typing import List, Optional
from openai import OpenAI
from src.config import KB_SUMMARY_FILE, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from src.utils.prompts import (
    GENERATE_KB_SUMMARY_PROMPT,
    UPDATE_KB_SUMMARY_PROMPT
)


class KnowledgeBaseSummary:
    """知识库总结管理器"""

    def __init__(self):
        self.summary_file = KB_SUMMARY_FILE
        os.makedirs(os.path.dirname(self.summary_file), exist_ok=True)

        if not LLM_API_KEY:
            raise ValueError("LLM_API_KEY is not set. Please set OPENAI_API_KEY in your .env file.")

        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    def get_summary(self) -> str:
        """读取知识库总结"""
        if os.path.exists(self.summary_file):
            with open(self.summary_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        return "知识库目前为空。"

    def generate_document_summary(self, title: str, content: str) -> Optional[str]:
        """生成单个文档的总结"""
        try:
            prompt = GENERATE_KB_SUMMARY_PROMPT.format(
                title=title,
                content=content[:3000]
            )

            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": "你是一个专业的文档总结助手。"}, {"role": "user", "content": prompt}],
                max_tokens=500
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"生成文档总结时出错: {e}")
            return None

    def update_summary(self, new_document_summaries: List[str]) -> bool:
        """更新知识库总结"""
        try:
            existing_summary = self.get_summary()
            combined_new_summary = "\n\n".join(new_document_summaries)

            if existing_summary == "知识库目前为空。":
                updated_summary = f"# 知识库内容概要\n\n{combined_new_summary}"
            else:
                prompt = UPDATE_KB_SUMMARY_PROMPT.format(
                    existing_summary=existing_summary,
                    new_summary=combined_new_summary
                )

                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "system", "content": "你是一个专业的文档总结助手。"}, {"role": "user", "content": prompt}],
                    max_tokens=1500
                )

                updated_summary = response.choices[0].message.content.strip()

            with open(self.summary_file, "w", encoding="utf-8") as f:
                f.write(updated_summary)

            print("✓ 知识库总结已更新", flush=True)
            return True

        except Exception as e:
            print(f"更新知识库总结时出错: {e}")
            return False

    def clear_summary(self) -> None:
        """清空知识库总结"""
        with open(self.summary_file, "w", encoding="utf-8") as f:
            f.write("知识库目前为空。")
        print("✓ 知识库总结已清空", flush=True)
