import torch
import transformers
from .base import LLM

class Llama3(LLM):
    
    def __init__(self, **kwargs):
        super().__init__(model_id="meta-llama/Meta-Llama-3-8B-Instruct", **kwargs)
    
    def load_model(self, token):
        print("Loading Llama3 Model")
        
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            token=token,
            model_kwargs={"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True},
            device_map="auto",
        )

        print("Loaded Llama3 Model")
    
    def invoke(self, input):
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": input},
        ]
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=1000,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        return outputs[0]["generated_text"][-1]["content"]