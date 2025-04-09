import requests
import html2text
from readability import Document
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import (BaseModel, Field)
from langchain_text_splitters import RecursiveCharacterTextSplitter

class FetchPageInput(BaseModel):
    url: str = Field()
    page_num: int = Field(0)


@tool(args_schema=FetchPageInput)
def fetch_page(url, page_num=0, timeout_sec=10):
    """
    指定されたURLから（とページ番号から）ウェブページのコンテンツを取得するツール。

    `status` と `page_content`（`title`、`content`、`has_next`インジケーター）を返します。
    statusが200でない場合は、ページの取得時にエラーが発生しています。（他のページの取得を試みてください）

    デフォルトでは、最大2,000トークンのコンテンツのみが取得されます。
    ページにさらにコンテンツがある場合、`has_next`の値はTrueになります。
    続きを読むには、同じURLで`page_num`パラメータをインクリメントして、再度入力してください。
    （ページングは0から始まるので、次のページは1です）

    1ページが長すぎる場合は、**3回以上取得しないでください**（メモリの負荷がかかるため）。

    Returns
    -------
    Dict[str, Any]:
    - status: str
    - page_content
      - title: str
      - content: str
      - has_next: bool
    """
    response = requests.get(url, timeout=timeout_sec)
    response.encoding = 'utf-8'

    doc = Document(response.text)
    title = doc.title()
    html_content = doc.summary()
    content = html2text.html2text(html_content)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name='gpt-3.5-turbo',
        chunk_size=1000,
        chunk_overlap=0,
    )
    chunks = text_splitter.split_text(content)
    return {
        "status": 200,
        "page_cotent":{
            "title": title,
            "content": chunks[page_num],
            "has_next": page_num < len(chunks)-1
        }   
    }

     