import torch, traceback
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
mid = "Qwen/Qwen2.5-VL-3B-Instruct"
bnb = BitsAndBytesConfig(load_in_8bit=True)

print("try load 8bit")
try:
    m = AutoModelForImageTextToText.from_pretrained(
        mid,
        quantization_config=bnb,
        trust_remote_code=True,
    )
    print("loaded", type(m))
except Exception as e:
    print("ERR", e)
    traceback.print_exc()
