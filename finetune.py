"""
Finetune GreesyGPT-Ultra on GreesyGuard moderation data.

Ultra is ~14x Mini's parameter count by default (see config.py), so
unlike a Mini finetune this is written to assume you may need:
  - multi-GPU DDP
  - gradient checkpointing
  - bf16/fp16 autocast + grad scaling
  - gradient accumulation to hit a reasonable effective batch size
  - periodic checkpointing (Ultra runs are expensive to lose)

Single GPU:
    python finetune.py --hf_dataset OnlyCheeini/greesyguard-3-mini-claude-4.6-sonnet-2000x \
        --tokenizer_json /path/to/tokenizer.json --output_dir ./ckpt-ultra

Multi-GPU (DDP), e.g. 4 GPUs on one node:
    torchrun --nproc_per_node=4 finetune.py \
        --hf_dataset OnlyCheeini/greesyguard-3-mini-claude-4.6-sonnet-2000x \
        --tokenizer_json /path/to/tokenizer.json --output_dir ./ckpt-ultra \
        --batch_size 2 --grad_accum_steps 16

If your GPUs can't fit the full ULTRA_CONFIG, pass --config ultra_small
for the ~4x-Mini middle-ground config instead of the ~14x one.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from config import ULTRA_CONFIG, ULTRA_SMALL_CONFIG, MINI_CONFIG
from model import GreesyGPTUltra
from tokenizer import GreesyTokenizer
from data import build_dataset, PadCollator

CONFIGS = {"ultra": ULTRA_CONFIG, "ultra_small": ULTRA_SMALL_CONFIG, "mini": MINI_CONFIG}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIGS.keys()), default="ultra")
    p.add_argument("--tokenizer_json", type=str, default=None,
                    help="Path to the GreesyGuard tokenizer.json. Required unless --smoke_test.")
    p.add_argument("--smoke_test", action="store_true",
                    help="Use the tiktoken o200k_base fallback tokenizer to sanity-check the "
                         "training loop only — NOT for real finetuning.")

    p.add_argument("--jsonl_path", type=str, default=None)
    p.add_argument("--hf_dataset", type=str, default=None,
                    help='e.g. "OnlyCheeini/greesyguard-3-mini-claude-4.6-sonnet-2000x"')
    p.add_argument("--hf_split", type=str, default="train")

    p.add_argument("--init_checkpoint", type=str, default=None,
                    help="Optional Ultra-sized checkpoint to warm-start from "
                         "(NOTE: cannot load Mini weights directly — different dims).")
    p.add_argument("--output_dir", type=str, default="./ckpt-ultra")
    p.add_argument("--resume_from", type=str, default=None)

    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1, help="per-GPU micro-batch size")
    p.add_argument("--grad_accum_steps", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_len", type=int, default=None,
                    help="Defaults to the config's context_len.")

    p.add_argument("--grad_checkpointing", action="store_true", default=True)
    p.add_argument("--no_grad_checkpointing", dest="grad_checkpointing", action="store_false")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")

    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def is_main(rank):
    return rank == 0


def cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr_ratio):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cos)


def enable_grad_checkpointing(model: GreesyGPTUltra):
    """Wraps each transformer block's forward in torch.utils.checkpoint."""
    for block in model.blocks:
        orig_forward = block.forward

        def make_ckpt_forward(fwd):
            def ckpt_forward(x, cos, sin, past_kv=None, use_cache=False):
                if past_kv is not None or use_cache or not x.requires_grad:
                    # cache-based decoding / eval — skip checkpointing
                    return fwd(x, cos, sin, past_kv, use_cache)
                return grad_checkpoint(
                    lambda x_: fwd(x_, cos, sin, None, False), x, use_reentrant=False
                )
            return ckpt_forward

        block.forward = make_ckpt_forward(orig_forward)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rank, world_size, local_rank, distributed = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    cfg = CONFIGS[args.config]
    max_len = args.max_len or cfg.context_len

    if is_main(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        n_params = cfg.approx_param_count()
        print(f"[config] {cfg.name}  (~{n_params/1e9:.2f}B params, {n_params/1e6:.0f}M)")
        print(f"[config] d_model={cfg.d_model} layers={cfg.n_layers} "
              f"heads={cfg.n_heads}/{cfg.n_kv_heads}kv context={cfg.context_len}")

    # ---- tokenizer -------------------------------------------------------
    if args.smoke_test:
        tok = GreesyTokenizer(use_tiktoken_fallback=True)
    else:
        if not args.tokenizer_json:
            raise ValueError("--tokenizer_json is required (or pass --smoke_test).")
        tok = GreesyTokenizer(tokenizer_json_path=args.tokenizer_json)

    # ---- data --------------------------------------------------------------
    train_ds, val_ds = build_dataset(
        tok, jsonl_path=args.jsonl_path, hf_dataset=args.hf_dataset,
        hf_split=args.hf_split, max_len=max_len, seed=args.seed,
    )
    pad_id = tok.token_to_id("<|endoftext|>")
    collator = PadCollator(pad_token_id=pad_id)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) \
        if distributed else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=collator,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    if is_main(rank):
        print(f"[data] train={len(train_ds)} val={len(val_ds)} "
              f"effective_batch={args.batch_size * args.grad_accum_steps * world_size}")

    # ---- model -------------------------------------------------------------
    model = GreesyGPTUltra(cfg).to(device)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        if is_main(rank):
            print(f"[init] loaded weights from {args.init_checkpoint}")

    if args.grad_checkpointing:
        enable_grad_checkpointing(model)

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    # ---- optimizer -----------------------------------------------------
    decay, no_decay = [], []
    raw_model = model.module if distributed else model
    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps = steps_per_epoch * args.epochs

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == "fp16"))

    start_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        if is_main(rank):
            print(f"[resume] resuming from step {start_step}")

    # ---- train loop ----------------------------------------------------
    model.train()
    step = start_step
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    running_loss = 0.0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                                 dtype=amp_dtype, enabled=(args.precision != "fp32")):
                _, loss, _ = model(input_ids, labels=labels)
                loss = loss / args.grad_accum_steps

            if args.precision == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item()
            micro_step += 1

            if micro_step % args.grad_accum_steps == 0:
                lr = cosine_lr(step, total_steps, args.warmup_steps, args.lr, args.min_lr_ratio)
                for g in optimizer.param_groups:
                    g["lr"] = lr

                if args.precision == "fp16":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)

                if args.precision == "fp16":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if is_main(rank) and step % args.log_every == 0:
                    dt = time.time() - t0
                    toks = args.batch_size * args.grad_accum_steps * world_size * input_ids.shape[1] * args.log_every
                    print(f"step {step:6d}/{total_steps}  loss {running_loss/args.log_every:.4f}  "
                          f"lr {lr:.2e}  {toks/dt:,.0f} tok/s")
                    running_loss = 0.0
                    t0 = time.time()

                if is_main(rank) and step % args.eval_every == 0:
                    evaluate(model, val_loader, device, amp_dtype, args.precision)
                    model.train()

                if is_main(rank) and step % args.save_every == 0:
                    save_checkpoint(raw_model, optimizer, step, args.output_dir)

            if step >= total_steps:
                break
        if step >= total_steps:
            break

    if is_main(rank):
        save_checkpoint(raw_model, optimizer, step, args.output_dir, final=True)
    if distributed:
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(model, val_loader, device, amp_dtype, precision, max_batches: int = 20):
    model.eval()
    losses = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                             dtype=amp_dtype, enabled=(precision != "fp32")):
            _, loss, _ = model(input_ids, labels=labels)
        losses.append(loss.item())
    if losses:
        avg = sum(losses) / len(losses)
        print(f"[eval] loss {avg:.4f}  ppl {math.exp(min(avg, 20)):.2f}")


def save_checkpoint(model, optimizer, step, output_dir, final: bool = False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    name = "final.pt" if final else f"step_{step}.pt"
    path = os.path.join(output_dir, name)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)
    print(f"[checkpoint] saved {path}")


if __name__ == "__main__":
    main()
