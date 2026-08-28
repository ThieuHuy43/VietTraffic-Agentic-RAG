import pdfplumber
import pytesseract
from typing import List, Dict, Any, Optional
from .base import BaseParser
from .utils import LegalStateTracker

class PdfParser(BaseParser):
    """
    Parser dành cho file PDF. Dùng pdfplumber để lấy text layer.
    Nếu không có text layer (pdf scan), fallback sang pytesseract OCR.
    Bảng (VD: bảng kích thước/màu sắc biển báo trong QCVN 41) được tách riêng bằng
    extract_tables() và loại khỏi vùng text thường (outside_bbox), tránh bị extract_text()
    làm rối thứ tự cột/hàng hoặc bị nhân đôi nội dung.
    """

    def _extract_cell_text(self, page, bbox) -> Optional[str]:
        """Trích text của 1 ô bảng theo bbox riêng (thay vì dùng text ghép sẵn của cả bảng),
        để phát hiện được ô nào chứa text xoay dọc (nhãn nhóm cột, VD "Biển bát giác" viết dọc
        trong bảng kích thước QCVN 41). pdfplumber gắn cờ upright=False cho ký tự bị xoay; khi
        phần lớn ký tự trong ô bị xoay, line-grouping mặc định của extract_text() không hiểu
        đúng chiều đọc của text xoay 90 độ nên trả ra thứ tự sai/lẫn nhiều từ với nhau.

        Đã xác minh thực tế trên PDF gốc: mỗi từ trong ô xoay nằm dọc theo 1 cột toạ độ x
        riêng (các từ xếp cạnh nhau trái->phải theo đúng chiều đọc), còn trong từng cột, ký tự
        đọc theo "top" (khoảng cách từ đỉnh trang) giảm dần mới đúng chiều đọc. Nên nhóm ký tự
        theo x0, đọc các nhóm trái->phải, và trong mỗi nhóm sắp theo top giảm dần."""
        if bbox is None:
            return None
        cropped = page.crop(bbox)
        chars = cropped.chars
        if not chars:
            return cropped.extract_text()

        non_upright = sum(1 for c in chars if not c.get("upright", True))
        if non_upright <= len(chars) / 2:
            return cropped.extract_text()

        groups: Dict[float, List[Dict[str, Any]]] = {}
        for c in chars:
            groups.setdefault(round(c["x0"]), []).append(c)

        parts = []
        for key in sorted(groups.keys()):
            group_chars = sorted(groups[key], key=lambda c: -c["top"])
            parts.append("".join(ch.get("text", "") for ch in group_chars))
        return "".join(parts)

    def _extract_table_rows(self, page, table) -> List[List[Optional[str]]]:
        return [
            [self._extract_cell_text(page, bbox) for bbox in row.cells]
            for row in table.rows
        ]

    def _table_to_text(self, rows: List[List[Optional[str]]]) -> Optional[str]:
        """Chuyển 1 bảng thô (list rows) thành text dạng "Header: value; Header2: value2"
        theo từng hàng, dùng hàng đầu làm header. Forward-fill các ô None ở đầu bảng
        (do cell bị merge/rowspan) bằng giá trị gần nhất cùng cột, để không mất ngữ cảnh
        khi 1 hàng chỉ có 1-2 cột thay đổi so với hàng trước."""
        if not rows or len(rows) < 2:
            return None

        def clean(cell: Optional[str]) -> str:
            return (cell or "").replace("\n", " ").strip()

        header = [clean(h) for h in rows[0]]
        if not any(header):
            return None

        lines = []
        last_values: List[Optional[str]] = [None] * len(header)
        for row in rows[1:]:
            cells = []
            for idx in range(len(header)):
                val = clean(row[idx]) if idx < len(row) else ""
                if not val and last_values[idx]:
                    val = last_values[idx]
                if val:
                    last_values[idx] = val
                cells.append(val)

            pairs = [f"{h}: {c}" for h, c in zip(header, cells) if h and c]
            if pairs:
                lines.append("; ".join(pairs))

        if not lines:
            return None
        return "\n".join(lines)

    def parse(self, file_path: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        tracker = LegalStateTracker(metadata)

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.find_tables()

                # Loại vùng bbox của các bảng khỏi trang trước khi extract_text() thường,
                # để nội dung bảng không lẫn vào text (méo thứ tự cột/hàng) hoặc bị lặp lại.
                text_page = page
                for table in tables:
                    text_page = text_page.outside_bbox(table.bbox)

                full_text = page.extract_text()
                if full_text and full_text.strip():
                    text = text_page.extract_text()
                    if text and text.strip():
                        for line in text.split('\n'):
                            tracker.process_line(line)
                else:
                    # Trang không có text layer nào (kể cả ngoài bảng) -> fallback OCR
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

                for table in tables:
                    table_text = self._table_to_text(self._extract_table_rows(page, table))
                    if table_text:
                        tracker.add_table_chunk(table_text)

        tracker.flush()
        return tracker.chunks
