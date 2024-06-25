import os
import uuid
from pathlib import Path
import faiss
import argparse
from dotenv import load_dotenv
from utils.ocr import OCRUtil
from vectorstores.faiss import MyFaissVectorStore, Document

if __name__ == "__main__":
    
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, type=str, help="The file to extract the prompt from")
    parser.add_argument('--classify', required=True, type=str, help="Classification of the file")

    args = parser.parse_args()
    
    ocr_util = OCRUtil()
    
    try:
        contents = ocr_util.extract_content(args.file)
    except ValueError as e:
        print(e)
        exit(1)
    
    
    documents = [
        Document(text=content, metadata={"is_extracapsular_nodal_extension_present": args.classify, "filename": args.file}, id_=str(uuid.uuid4()))
        for content in contents
    ]
    
    faiss_vector_store = MyFaissVectorStore()
    faiss_vector_store.add_document(documents)