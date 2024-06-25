import os
import csv
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from prompts import Prompts
from llms.bloom import Blooom
from llms.tinyllama import TinyLlama
from llms.llama3 import Llama3
from vectorstores.faiss import MyFaissVectorStore
from utils.ocr import OCRUtil

if __name__ == "__main__":
    
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=str, help="The model checkpoint to use")
    parser.add_argument('--file', required=True, type=str, help="The file to extract the prompt from")
    parser.add_argument('--output', required=False, type=str, help="CSV file to add result", default="./result.csv")

    args = parser.parse_args()
        
    model_class = {
        "bloom": Blooom,
        "tinyllama": TinyLlama,
        "llama3": Llama3,
    }.get(args.checkpoint, None)

    if model_class is None:
        print("Not supported model")
        exit(1)
        
    model = model_class(system_prompt=Prompts.is_extracapsular_nodal_extension_present(""))
    if args.checkpoint == "llama3":
        model.load_model(token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"))
    else:
        model.load_model()
    
    
    ocr_util = OCRUtil()
    
    try:
        contents = ocr_util.extract_content(args.file)
    except ValueError as e:
        print(e)
        exit(1)
    
    index = 1
    for content in contents:
        vector_store = MyFaissVectorStore()
        
        vectors = vector_store.retreive(query=content)
        
        final_context_str = ""
        for vector in vectors:
            final_context_str = final_context_str + """            
--------------------------------------------------
Extracapsular Nodal Extension Presence: {presence}

Reports:
{report}
--------------------------------------------------
""".format(presence=vector.metadata["is_extracapsular_nodal_extension_present"], report=vector.text)
        
        model.system_prompt = Prompts.is_extracapsular_nodal_extension_present(final_context_str)
        print(f"Example contexts: \n\n {final_context_str}")
        
        output = model.invoke(content)
        output_dict = json.loads(output)
        
        print(f"Result {index}:", output_dict)
        
        # Define the path to the CSV file
        csv_file_path = Path(args.output)
        
        # Check if the file exists
        file_exists = csv_file_path.is_file()

        # Open the CSV file in append mode ("a" mode)
        with open(csv_file_path, mode='a', newline='') as file:
            output_dict["file"] = args.file
            writer = csv.DictWriter(file, fieldnames=output_dict.keys())

            # If the file does not exist, write the header first
            if not file_exists:
                writer.writeheader()

            # Write the dictionary as a new row
            writer.writerow(output_dict)

        print(f"Saved to {csv_file_path}.")