from bs4 import BeautifulSoup
from typing import List, Dict, Any
from .base import BaseParser
from .utils import LegalStateTracker

class HtmlParser(BaseParser):
    """Parser dành cho các file HTML. Loại bỏ tag nav/header/footer rác."""
    def parse(self, file_path: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            
        # Loại bỏ các tag rác
        for tag in soup(["header", "footer", "nav", "script", "style", "aside"]):
            tag.decompose()
            
        tracker = LegalStateTracker(metadata)
        
        # Duyệt qua các block chứa chữ
        for element in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
            # Dùng separator khoảng trắng để tránh dính chữ
            text = element.get_text(separator=" ", strip=True)
            if text:
                tracker.process_line(text)
                
        tracker.flush()
        return tracker.chunks
