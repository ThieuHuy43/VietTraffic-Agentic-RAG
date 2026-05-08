from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseParser(ABC):
    """
    Base class cho các parser văn bản pháp luật.
    Mọi parser cụ thể (HTML, DOCX, PDF) đều phải kế thừa class này
    và implement phương thức parse().
    """
    
    @abstractmethod
    def parse(self, file_path: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse file văn bản pháp luật và trả về list các dictionary chunk theo schema bắt buộc.
        
        metadata bao gồm:
        - doc_id: str
        - doc_type: str
        - status: str
        - effective_date: str
        - amends: str | None
        - supersedes: str | None
        """
        pass
