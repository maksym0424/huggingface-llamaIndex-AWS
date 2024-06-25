import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import LLM

class Blooom(LLM):
    
    def __init__(self, **kwargs):
        super().__init__(model_id="bigscience/bloomz-7b1", **kwargs)
    
    def load_model(self):
        print("Loading Bloom Model")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
        print("Loaded Bloom Model")
    
    def invoke(self, input):
        inputs = self.tokenizer.encode(input, return_tensors="pt").to("cuda")
        outputs = self.model.generate(inputs, max_new_tokens=1000)
        output_text = self.tokenizer.decode(outputs[0][inputs.size(1)], skip_special_tokens=True)
        
        return output_text