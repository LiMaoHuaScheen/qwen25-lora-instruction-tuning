import argparse, json, math, os, time, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

def set_seed(s=42):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def format_chat(tokenizer, ex):
    if "instruction" in ex and "output" in ex:
        inp = ex["instruction"]; out = ex["output"]
        if "input" in ex and ex["input"] and str(ex["input"]).strip() and str(ex["input"]) != "<noinput>":
            inp = inp + "\n" + str(ex["input"])
    elif "prompt" in ex and "response" in ex:
        inp = ex["prompt"]; out = ex["response"]
    else:
        inp = ex.get("instruction", ex.get("prompt", "")); out = ex.get("output", ex.get("response", ""))
    msgs = [{"role":"user","content":inp},{"role":"assistant","content":out}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

class InstructDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data; self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        ex = self.data[i]
        text = format_chat(self.tokenizer, ex)
        enc = self.tokenizer(text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"][0]
        labels = input_ids.clone()
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        labels[input_ids == pad_id] = -100
        return {"input_ids": input_ids, "attention_mask": enc["attention_mask"][0], "labels": labels}

def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

def evaluate(model, tokenizer, dataloader, device):
    model.eval(); total_loss = 0; n = 0
    with torch.no_grad():
        for b in dataloader:
            input_ids = b["input_ids"].to(device); attn = b["attention_mask"].to(device); labels = b["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            total_loss += out.loss.item() * input_ids.size(0)
            n += input_ids.size(0)
    return total_loss / max(n, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="PROJECT_ROOT/../hf_cache/Qwen2.5-1.5B")
    ap.add_argument("--data", default="PROJECT_ROOT/datasets/alpaca_gpt4_data_zh.json")
    ap.add_argument("--output", default="PROJECT_ROOT/output_lora")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--mode", default="lora", choices=["zero", "lora", "full"])
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    set_seed(42)
    device = args.device if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dt).to(device)
    model.config.use_cache = False
    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model
        lora = LoraConfig(r=args.r, lora_alpha=args.r*2, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
    data = json.load(open(args.data, encoding="utf-8"))
    if args.limit > 0: data = data[:args.limit]
    random.shuffle(data)
    split = int(len(data)*0.9)
    train_ds = InstructDataset(data[:split], tokenizer, args.max_len)
    eval_ds = InstructDataset(data[split:], tokenizer, args.max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)
    if args.mode != "zero":
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        total_steps = (len(train_dl)//args.grad_accum)*args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1,int(total_steps*0.03)), num_training_steps=total_steps)
        scaler = torch.cuda.amp.GradScaler() if args.dtype == "fp16" else None
        history = {"train_loss": [], "eval_loss": []}
        step = 0
        for ep in range(args.epochs):
            model.train(); ep_loss = 0; ep_n = 0
            optimizer.zero_grad()
            for b in train_dl:
                input_ids = b["input_ids"].to(device); attn = b["attention_mask"].to(device); labels = b["labels"].to(device)
                with torch.autocast(device_type="cuda", dtype=dt, enabled=(args.dtype=="fp16")):
                    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = out.loss / args.grad_accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                ep_loss += out.loss.item(); ep_n += 1
                if (ep_n % args.grad_accum) == 0:
                    params = [p for p in model.parameters() if p.requires_grad]
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                    else:
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        optimizer.step(); scheduler.step(); optimizer.zero_grad()
                    step += 1
                if step % 20 == 0:
                    import io as _io
                    with _io.open(os.path.join(args.output, "progress.txt"), "w", encoding="utf-8") as _pf:
                        _pf.write(f"step={step} ep={ep+1} loss={out.loss.item():.4f}\n")
                if args.max_steps and step >= args.max_steps: break
            if args.max_steps and step >= args.max_steps: break
            ev = evaluate(model, tokenizer, eval_dl, device)
            history["train_loss"].append(ep_loss/max(ep_n,1)); history["eval_loss"].append(ev)
            print(f"epoch {ep+1} train_loss={ep_loss/max(ep_n,1):.4f} eval_loss={ev:.4f}")
        json.dump(history, open(os.path.join(args.output, "history.json"), "w", encoding="utf-8"))
        if args.mode == "lora":
            model.save_pretrained(os.path.join(args.output, "lora_weights"))
        else:
            torch.save(model.state_dict(), os.path.join(args.output, "full_weights.pt"))
    else:
        ev = evaluate(model, tokenizer, eval_dl, device)
        print(f"zero-shot eval_loss={ev:.4f} ppl={math.exp(ev):.2f}")
        json.dump({"eval_loss": ev, "ppl": math.exp(ev)}, open(os.path.join(args.output, "zero_result.json"), "w", encoding="utf-8"))
    samples = []
    for ex in random.sample(data[split:], min(5, len(data)-split)):
        if "instruction" in ex:
            inp = ex["instruction"] + (("\n"+str(ex["input"])) if ex.get("input") and str(ex["input"]).strip() and str(ex["input"])!="<noinput>" else "")
        else:
            inp = ex.get("prompt","")
        msgs = [{"role":"user","content":inp}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        gen = model.generate(ids, max_new_tokens=100, do_sample=False)
        out_text = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
        samples.append({"prompt": inp, "reference": ex.get("output", ex.get("response","")), "generated": out_text})
    json.dump(samples, open(os.path.join(args.output, "samples.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()