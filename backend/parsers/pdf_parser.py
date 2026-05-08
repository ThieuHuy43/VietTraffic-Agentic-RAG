import pdfplumber
import pytesseract
from typing import List, Dict, Any
from .base import BaseParser
from .utils import LegalStateTracker

class PdfParser(BaseParser):
    """
    Parser dành cho file PDF. Dùng pdfplumber để lấy text layer.
    Nếu không có text layer (pdf scan), fallback sang pytesseract OCR.
    """
    def parse(self, file_path: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        tracker = LegalStateTracker(metadata)
        
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    for line in text.split('\n'):
                        tracker.process_line(line)
                else:
                    # Fallback to OCR if no text found in page
                    img = page.to_image(resolution=200).original
                    try:
                        # Giả định hệ thống đã cài Tesseract và gói ngôn ngữ tiếng Việt (vie)
                        ocr_text = pytesseract.image_to_string(img, lang='vie')
                        if ocr_text and ocr_text.strip():
                            for line in ocr_text.split('\n'):
                                tracker.process_line(line)
                    except Exception as e:
                        # Bỏ qua nếu OCR thất bại
                        pass
                        
        tracker.flush()
        return tracker.chunks
