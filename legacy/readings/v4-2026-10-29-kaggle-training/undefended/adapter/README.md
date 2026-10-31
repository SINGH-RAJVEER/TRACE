---
base_model: Qwen/Qwen3.5-2B-Base
library_name: peft
model_name: adapter
tags:
- base_model:adapter:Qwen/Qwen3.5-2B-Base
- lora
- sft
- transformers
- trl
licence: license
pipeline_tag: text-generation
---

# Model Card for adapter

This model is a fine-tuned version of [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device_map="auto")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 



This model was trained with SFT.

### Framework versions

- PEFT 0.21.2
- TRL: 1.15.0
- Transformers: 5.19.0
- Pytorch: 2.14.1
- Datasets: 5.0.1
- Tokenizers: 0.23.3

## Citations



Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```