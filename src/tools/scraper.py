"""
网页下载工具 - 使用 requests + BeautifulSoup
备用方案：使用 jina.ai 抓取服务处理受保护的网站
"""
import os
from typing import Optional
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import markdownify


def _download_with_jina(url: str, filepath: str) -> bool:
    """使用 jina.ai 抓取服务作为备用方案"""
    try:
        jina_url = f"https://r.jina.ai/http://{url.replace('https://', '').replace('http://', '')}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        response = requests.get(jina_url, headers=headers, timeout=30)
        response.raise_for_status()

        # jina.ai 返回纯文本，直接保存
        content = response.text.strip()
        if content and len(content) > 100:
            full_content = f"# {url}\n\n{content}"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(full_content)
            print(f"✓ 下载完成 (使用备用方案): {filepath}")
            return True
        return False
    except Exception:
        return False


def download_page(url: str, output_dir: str = "data/documents") -> Optional[str]:
    """
    使用 requests 下载网页内容

    Args:
        url: 要下载的网页 URL
        output_dir: 保存目录

    Returns:
        保存的文件路径，失败时返回 None
    """
    os.makedirs(output_dir, exist_ok=True)

    # 从 URL 生成文件名
    parsed_url = urlparse(url)
    filename = parsed_url.path.replace("/", "_").strip("_")
    if not filename:
        filename = "index"
    filename = f"{filename}.md"
    filepath = os.path.join(output_dir, filename)

    try:
        # 发送请求 - 使用更完整的请求头
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        response.encoding = response.apparent_encoding

        # 解析 HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # 移除不需要的标签
        for script in soup(["script", "style", "nav", "footer", "header", "iframe"]):
            script.decompose()

        # 尝试找到主要内容区域
        main_content = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", class_="content") or
            soup.find("div", id="content") or
            soup.find("div", class_="post") or
            soup.find("div", class_="article") or
            soup.body
        )

        if main_content:
            # 转换为 Markdown
            markdown_content = markdownify.markdownify(
                str(main_content),
                heading_style="ATX"
            )

            # 清理多余的空行
            lines = markdown_content.split('\n')
            cleaned_lines = []
            prev_empty = False
            for line in lines:
                if line.strip():
                    cleaned_lines.append(line)
                    prev_empty = False
                elif not prev_empty:
                    cleaned_lines.append(line)
                    prev_empty = True

            markdown_content = '\n'.join(cleaned_lines)

            # 添加元数据
            full_content = f"# {url}\n\n{markdown_content}"

            # 保存为 Markdown 文件
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(full_content)

            print(f"✓ 下载完成: {filepath}")
            return filepath

        return None

    except Exception as e:
        # 尝试使用备用方案
        print(f"⚠️ 直接下载失败，尝试备用方案...")
        if _download_with_jina(url, filepath):
            return filepath
        print(f"✗ 下载失败 {url}: {e}")
        return None
