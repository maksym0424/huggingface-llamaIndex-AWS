import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from .base import LLM

class TinyLlama(LLM):
    
    def __init__(self, **kwargs):
        super().__init__(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", **kwargs)
    
    def load_model(self):
        print("Loading TinyLlama Model")
        self.pipeline = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
        print("Loaded TinyLlama Model")
    
    def invoke(self, input):
        messages = [
            {
                "role": "system",
                "content": "You are an Assistant.",
            },
            {"role": "user", "content": input},
        ]
        prompt = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipeline(prompt, max_new_tokens=1000, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        
        return outputs[0]["generated_text"][len(prompt):]