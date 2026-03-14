import os
from duckduckgo_search import DDGS
import json
import requests
from typing import List, Literal, Optional, Union
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
load_dotenv()

# 使用免费的 duck duck go 进行网页检索，使用 firecrawl 将网页转化为markdown格式
def web_search_DDG(query: Optional[str], search_num: int = 2, search_mode: str = 'fast') -> Optional[List[str]]:
    assert search_mode == 'fast' or search_mode == 'pro'
    if search_mode == 'fast':
        assert type(query)==str
        results = DDGS().text(query, max_results=search_num)
        return results
    elif search_mode == 'pro':
        assert type(query)==str
        firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API"))
        results = DDGS().text(query, max_results=search_num)
        for result in results:
            web_url = result['href']
            # firecrawl_app returns markdown and metadata
            web_content = firecrawl_app.scrape_url(web_url)
            web_content_markdown = web_content['markdown']
            web_content_metadata = web_content['metadata']
            result['web_content_markdown'] = web_content_markdown
            result['web_content_metadata'] = web_content_metadata
        return results

def web_search_SERPER_API(query: Optional[str], search_num=2, search_mode: str = 'fast') -> Optional[List[str]]:
    assert search_mode == 'fast' or search_mode == 'pro'
    if search_mode == 'fast':
        assert type(query)==str
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": search_num})
        headers = {'X-API-KEY': os.getenv('SERPER_API'), 'Content-Type': 'application/json'}
        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        results = []
        for item in response['organic']:
            results.append(
                {'title': item['title'], 'href':item['link'], 'body': item['snippet']}
            )
        return results
    elif search_mode == 'pro':
        assert type(query)==str
        firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API"))
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": search_num})
        headers = {'X-API-KEY': os.getenv('SERPER_API'), 'Content-Type': 'application/json'}
        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        results = []
        for item in response['organic']:
            results.append(
                {'title': item['title'], 'href':item['link'], 'body': item['snippet']}
            )
        for result in results:
            web_url = result['href']
            # firecrawl_app returns markdown and metadata
            web_content = firecrawl_app.scrape_url(web_url)
            web_content_markdown = web_content['markdown']
            web_content_metadata = web_content['metadata']
            result['web_content_markdown'] = web_content_markdown
            result['web_content_metadata'] = web_content_metadata
        return results

def web_search_BOCHA_API(query: Optional[str], search_num=2, search_mode: str = 'fast') -> Optional[List[str]]:
    """使用博查 Search API 进行网页搜索。"""
    assert search_mode == 'fast' or search_mode == 'pro'
    assert type(query) == str

    api_key = os.getenv("BOCHA_API")
    if not api_key:
        raise ValueError("BOCHA_API environment variable is required")

    url = "https://api.bochaai.com/v1/web-search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "count": search_num,
        "summary": True,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()
    data = response.json()

    results = []
    web_pages = data.get("webPages", {}).get("value", [])
    for item in web_pages:
        results.append(
            {
                "title": item.get("name", ""),
                "href": item.get("url", ""),
                "body": item.get("summary", item.get("snippet", "")),
            }
        )

    if search_mode == 'pro':
        firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API"))
        for result in results:
            try:
                web_url = result['href']
                # firecrawl_app returns markdown and metadata
                web_content = firecrawl_app.scrape_url(web_url)
                web_content_markdown = web_content['markdown']
                web_content_metadata = web_content['metadata']
                result['web_content_markdown'] = web_content_markdown
                result['web_content_metadata'] = web_content_metadata
            except Exception as e:
                print(f"警告: 无法抓取URL {web_url}: {e}")
                result['web_content_markdown'] = ""
                result['web_content_metadata'] = {}

    return results