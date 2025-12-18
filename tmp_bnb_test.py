import torch, traceback
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
mid = "jarvisvasu/Qwen2.5-VL-3B-Instruct-4bit"
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("try load device_map none")
try:
    m = AutoModelForImageTextToText.from_pretrained(
        mid,
        quantization_config=bnb,
        device_map=None,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    )
    print("loaded", type(m))
except Exception as e:
    print("ERR", e)
    traceback.print_exc()

print("try load device_map cuda")
try:
    m2 = AutoModelForImageTextToText.from_pretrained(
        mid,
        quantization_config=bnb,
        device_map={"": 0},
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    )
    print("loaded2", type(m2))
except Exception as e:
    print("ERR2", e)
    traceback.print_exc()
