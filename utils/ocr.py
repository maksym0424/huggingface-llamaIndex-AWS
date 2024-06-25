import os
import io
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path

class OCRUtil:
    
    def __init__(self):
        pass
    
    def read_text_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()

    def read_image_file(self, file_path):
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)

    def read_pdf_file(self, file_path):
        reader = PdfReader(file_path)
        prompts = []
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            if text:
                prompts.append(text.strip())
            else:
                image = convert_from_path(file_path, first_page=page_num+1, last_page=page_num+1)[0]
                text = pytesseract.image_to_string(image)
                prompts.append(text.strip())
        return prompts
    
    def extract_content(self, file_path):
        """
        Extract content from the file path.
        
        Supported File Type: "txt", "jpg", "jpeg", "png", "pdf"
        
        Return:

            List[str]: extracted contents(each page content)
        """
        _, ext = os.path.splitext(file_path)
        if ext == '.txt':
            return [self.read_text_file(file_path)]
        elif ext in ['.jpg', '.jpeg', '.png']:
            return [self.read_image_file(file_path)]
        elif ext == '.pdf':
            return self.read_pdf_file(file_path)
        else:
            raise ValueError("Unsupported file type: use 'txt', 'jpg', 'jpeg', 'png' or 'pdf'")
    
    