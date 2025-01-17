from aiohttp_client_cache import CachedSession
import bs4
from tenacity import retry, stop_after_attempt, wait_exponential
from base64 import b64decode
from aiohttp import BasicAuth

class BaseParser:
    def __init__(self, url: str, session: CachedSession, zyte: bool = False) -> None:
        self.url = url
        self.soup = None
        self.session = session
        self.zyte = zyte
        self.object_id = url.split('/')[-1]


    # @retry(stop=stop_after_attempt(30), wait=wait_exponential(multiplier=1, min=4, max=40))
    async def fetch_page_contents(self, url):
        # TODO: Handling errors
        if self.zyte:
            api_response = await self.session.post(
                "https://api.zyte.com/v1/extract",
                auth=BasicAuth("23a11568431b477e982ad2384e003bf8", ""),
                json={
                "url": url,
                "httpResponseBody": True,
            },
            )
            response_json = await api_response.json()
            api_response.raise_for_status()
            http_response_body: str = b64decode(
                response_json["httpResponseBody"]).decode("utf-8")
            return http_response_body
        else:
            response = await self.session.get(url)
            response.raise_for_status()
            text = await response.text()
            return text
    
    
    async def get_soup(self):
        if self.soup is None:
            response_text = await self.fetch_page_contents(self.url)
            if not response_text:
                raise ValueError(f"Got empty response for {self.url}")
            self.soup = bs4.BeautifulSoup(response_text, 'html.parser')
        return self.soup