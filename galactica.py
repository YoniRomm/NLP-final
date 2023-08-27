# https://huggingface.co/GeorgiaTechResearchInstitute/galactica-30b-evol-instruct-70k

# pip install accelerate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("GeorgiaTechResearchInstitute/galactica-30b-evol-instruct-70k")
model = AutoModelForCausalLM.from_pretrained("GeorgiaTechResearchInstitute/galactica-30b-evol-instruct-70k", device_map="auto", torch_dtype=torch.bfloat16)

# the evol-instruct models were fine-tuned with the same hidden prompts as the Alpaca project
no_input_prompt_template = ("### Instruction:\n{instruction}\n\n### Response:")
prompt = "Write out Maxwell's equations and explain the meaning of each one."
formatted_prompt = no_input_prompt_template.format_map({'instruction': prompt})

tokenized_prompt = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
out_tokens = model.generate(tokenized_prompt)

print(tokenizer.batch_decode(out_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False))
