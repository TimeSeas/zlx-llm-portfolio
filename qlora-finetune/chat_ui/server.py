"""Qwen2.5 Chat Server — Flask backend with fine-tuned model"""
import os, sys, json, torch
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-1.5B-Instruct")
LORA_PATH = os.path.join(BASE_DIR, "output", "qwen2.5-lora", "final_lora")
USE_LORA = os.path.exists(LORA_PATH)

print("=" * 50)
print(" Qwen2.5 Chat Server")
print("=" * 50)

# Load model once at startup
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

print("\n[1/3] Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("[2/3] Loading model (4-bit)...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, torch_dtype=torch.float16,
    max_memory={0: "6GB", "cpu": "16GB"},
)

if USE_LORA:
    print(f"[3/3] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    print(f"  Fine-tuned model ready!")
else:
    print("[3/3] Using base model (no LoRA)")
model.eval()

print(f"\nServer ready at http://127.0.0.1:5000")
print("=" * 50)

# ==========================================
# Flask App
# ==========================================
from flask import Flask, request, jsonify, render_template, Response
import flask

app = Flask(__name__)

# Conversation history storage (in-memory, per-session basic)
sessions = {}

def generate_response(messages, max_tokens=512, temperature=0.7):
    """Stream or generate response from model"""
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tok.pad_token_id,
            repetition_penalty=1.05,
        )

    response = tok.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


@app.route("/")
def index():
    return render_template("index.html", use_lora=USE_LORA)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.7)

    if not messages:
        return jsonify({"error": "No messages"}), 400

    # Always prepend system message
    full_messages = [
        {"role": "system", "content": ""}
    ] + messages

    try:
        response = generate_response(full_messages, max_tokens, temperature)
        return jsonify({
            "response": response,
            "model": f"Qwen2.5-1.5B-Instruct {'+LoRA' if USE_LORA else '(base)'}",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "Qwen2.5-1.5B-Instruct",
        "lora": USE_LORA,
        "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
