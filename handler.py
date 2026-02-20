import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "/model"

print("Loading RoGPT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
print("RoGPT ready!")

def handler(job):
    inp      = job["input"]
    prompt   = inp.get("prompt", "")
    max_tok  = inp.get("max_tokens", 500)
    temp     = inp.get("temperature", 0.7)

    full = (
        f"<|im_start|>system\n"
        f"You are RoGPT, an expert Roblox Luau developer.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(full, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tok,
            temperature=temp,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return {"response": response}

runpod.serverless.start({"handler": handler})
