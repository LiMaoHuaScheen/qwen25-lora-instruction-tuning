import argparse, json, math, os, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def set_seed(s=42):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="PROJECT_ROOT/../hf_cache/Qwen2.5-1.5B")
    ap.add_argument("--lora", default="")
    ap.add_argument("--data", default="PROJECT_ROOT/datasets/alpaca_gpt4_data_zh.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--output", default="PROJECT_ROOT/output_lora/eval.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    set_seed(42)
    device = args.device if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
    if args.lora:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).to(device)
        model = PeftModel.from_pretrained(base, args.lora).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).to(device)
    model.eval()
    data = json.load(open(args.data, encoding="utf-8"))
    random.seed(42); random.shuffle(data)
    eval_data = data[-args.n_eval:] if args.n_eval else data
    total_ll = 0.0; total_tokens = 0
    with torch.no_grad():
        for ex in eval_data:
            inp = ex["instruction"] + (("\n"+str(ex["input"])) if ex.get("input") and str(ex["input"]).strip() and str(ex["input"])!="<noinput>" else "")
            msgs = [{"role":"user","content":inp},{"role":"assistant","content":ex["output"]}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            enc = tokenizer(text, return_tensors="pt", max_length=args.max_len, truncation=True)
            ids = enc.input_ids.to(device); attn = enc.attention_mask.to(device)
            out = model(input_ids=ids, attention_mask=attn, labels=ids)
            total_ll += out.loss.item() * ids.shape[1]
            total_tokens += ids.shape[1]
    ppl = math.exp(total_ll / max(total_tokens, 1))
    samples = []
    for ex in random.sample(eval_data, min(8, len(eval_data))):
        inp = ex["instruction"] + (("\n"+str(ex["input"])) if ex.get("input") and str(ex["input"]).strip() and str(ex["input"])!="<noinput>" else "")
        msgs = [{"role":"user","content":inp}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        gen = model.generate(ids, max_new_tokens=120, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        out_text = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
        samples.append({"prompt": inp, "reference": ex["output"], "generated": out_text})
    result = {"perplexity": ppl, "n_eval": len(eval_data), "samples": samples}
    json.dump(result, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"ppl={ppl:.2f}")

if __name__ == "__main__":
    main()
