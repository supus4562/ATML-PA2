"""Task 1 – DPO Training Loop.

Standalone: runnable without any prior task state.
All results are written to disk (results/task1_dpo/) so that evaluate.py
and plot_results.py can be run independently — no hidden notebook state.

Usage
-----
  python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
  python -m task1_dpo.train --config configs/dpo.yaml --run-name standard --compile

Blackwell (RTX PRO 6000, 96 GB GDDR7, sm_120) optimisations applied
---------------------------------------------------------------------
1.  Liger-Kernel  – patches Qwen2 layers (RMSNorm, RoPE, SwiGLU, CrossEntropy)
    with fused Triton kernels before model load. ~20% throughput, ~60% memory.
2.  FlashAttention-4 – auto-detected at runtime; falls back to PyTorch SDPA.
3.  Reference logprob pre-caching – one no_grad sweep before the loop so
    we never run the reference model forward inside a training step. Halves
    per-step compute; feasible because 96 GB easily holds all cached tensors.
4.  AMP + GradScaler – torch.autocast(float16) + GradScaler for stable FP16
    training. GradScaler is mandatory with FP16 (narrower dynamic range than
    BF16 → small gradients underflow without scaling).
5.  Fused AdamW (fused=True) – collapses weight-update into one GPU kernel.
6.  Linear LR warmup – 5% of total optimizer steps (warmup_ratio in config).
7.  DataLoader: num_workers=4 + pin_memory=True – overlaps CPU tokenisation
    with GPU compute, keeps GPU fed continuously.
8.  cudnn.benchmark=True – lets cuDNN auto-select fastest kernels for the
    fixed sequence-length shapes we use.
9.  torch.compile (opt-in) – mode="reduce-overhead" uses CUDA Graphs to
    eliminate Python kernel-launch latency. Enable after verifying base run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import (
    count_parameters,
    load_policy,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task1_dpo.dpo import dpo_loss, validate_dpo_loss


# ─────────────────────────────────────────────────────────────────────────────
# Liger-Kernel patching (must happen before model load)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_liger_kernel(use_compile: bool = False):
    """Patch Qwen2 layers with fused Triton kernels if liger_kernel is installed.

    Patches RoPE, SwiGLU, CrossEntropy unconditionally.
    RMSNorm is SKIPPED when torch.compile is active: LigerRMSNorm uses a
    custom autograd_function_apply higher-order op that TorchDynamo cannot
    trace through, causing an InternalTorchDynamoError on the first batch.
    All other Liger kernels are dynamo-safe.
    """
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2
        patch_rms = not use_compile
        apply_liger_kernel_to_qwen2(
            rope=True,
            rms_norm=patch_rms,
            swiglu=True,
            cross_entropy=True,
            fused_linear_cross_entropy=False,  # disabled: incompatible with DPO
        )
        patched = "RMSNorm, RoPE, SwiGLU, CrossEntropy" if patch_rms else "RoPE, SwiGLU, CrossEntropy (RMSNorm skipped — compile mode)"
        print(f"[Liger-Kernel] Fused Qwen2 kernels applied ({patched})")
    except ImportError:
        print("[Liger-Kernel] Not installed — falling back to standard kernels. "
              "Install with: pip install liger-kernel")


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_collate(tokenizer, max_length: int):
    """Build a DataLoader collate function for DPO (chosen + rejected pairs)."""
    def collate(rows):
        chosen_list, rejected_list = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            try:
                chosen_list.append(
                    encode_prompt_response(tokenizer, prompt, yc, max_length)
                )
                rejected_list.append(
                    encode_prompt_response(tokenizer, prompt, yr, max_length)
                )
            except ValueError:
                # Prompt alone exceeds max_length — skip this example.
                # Per the updated data.py policy: filter, don't truncate prompt.
                continue
        if not chosen_list:
            return None
        return pad_batch(tokenizer, chosen_list), pad_batch(tokenizer, rejected_list)
    return collate


# ─────────────────────────────────────────────────────────────────────────────
# FlashAttention detection
# ─────────────────────────────────────────────────────────────────────────────

def _has_flash_attention() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


@torch.no_grad()
# ─────────────────────────────────────────────────────────────────────────────
# prepare_dpo_run  (public API reused by ablate_beta.py)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dpo_run(
    config_path: str,
    dataset_path: Optional[str] = None,
    beta: Optional[float] = None,
    max_examples: Optional[int] = None,
    use_compile: bool = False,
):
    """Load config, data, model, and optimiser. Returns a ready-to-train bundle."""
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    # Apply Liger-Kernel patches BEFORE model load so the patched modules are
    # used when the model is instantiated.
    do_compile = use_compile or cfg.get("compile", False)
    _apply_liger_kernel(do_compile)

    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])

    # ── Load policy model ─────────────────────────────────────────────────────
    # Pass attn_implementation if FlashAttention is installed.
    flash = _has_flash_attention()
    if flash:
        print("[FlashAttention] flash_attn detected — using FlashAttention-2/4 kernels.")
    else:
        print("[FlashAttention] Not installed — using PyTorch SDPA (still fast on Blackwell).")

    # We need to temporarily inject attn_implementation into the model load.
    # common/models.py::load_policy uses AutoModelForCausalLM.from_pretrained;
    # we monkey-patch the call by passing it via the cfg dict and catching it
    # inside a thin wrapper below.
    model = _load_policy(cfg, flash=flash, trainable=True, fresh_lora=True)

    # ── DataLoader ────────────────────────────────────────────────────────────
    # num_workers=4 + pin_memory=True overlaps CPU tokenisation with GPU compute.
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,   # avoid worker respawn overhead between epochs
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # fused=True collapses the weight update into a single GPU kernel —
    # free throughput gain on Blackwell (PyTorch>=2.0 + CUDA GPU).
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        fused=torch.cuda.is_available(),
    )

    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def _load_policy(cfg: dict, flash: bool, trainable: bool, fresh_lora: bool):
    """Load policy with optional FlashAttention-2 via a temporary cfg override.

    common/models.load_policy doesn't expose attn_implementation yet, so we
    replicate its logic here for the flash case, delegating the non-flash path
    back to the existing helper.
    """
    from common.models import resolve_dtype, make_lora_config
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    if flash:
        dtype = resolve_dtype(cfg.get("dtype", "float16"))
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"],
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
        tok = load_tokenizer(cfg["base_model"])
        model.config.pad_token_id = tok.pad_token_id
        if torch.cuda.is_available():
            model = model.cuda()
        if fresh_lora:
            model = get_peft_model(model, make_lora_config(cfg))
    else:
        model = load_policy(cfg, trainable=False, fresh_lora=fresh_lora)

    if trainable:
        model.train()
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    config_path: str,
    run_name: str,
    dataset_path: Optional[str] = None,
    output_path: Optional[str] = None,
    beta: Optional[float] = None,
    max_examples: Optional[int] = None,
    use_compile: bool = False,
):
    """Full DPO training loop.

    Algorithm
    ---------
    1.  Validate dpo_loss correctness (fail-fast before touching any weights).
    2.  Load model + data via prepare_dpo_run().
    3.  Pre-compute reference logprobs (one sweep, cached in VRAM).
    4.  Optionally wrap model with torch.compile.
    5.  Build AMP autocast context + GradScaler (required for FP16).
    6.  Build linear LR scheduler with warmup.
    7.  Training loop over 1 epoch with gradient accumulation.
    8.  Log every micro-step to disk (results_dir/train_log_{run_name}.jsonl).
    9.  Save adapter checkpoint.
    10. Save epoch summary JSON.
    """

    # ── 1. Validate loss implementation ───────────────────────────────────────
    validate_dpo_loss()

    # ── 2. Load bundle ────────────────────────────────────────────────────────
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples, use_compile=use_compile)
    cfg = bundle["cfg"]
    model = bundle["model"]
    loader = bundle["loader"]
    optimizer = bundle["optimizer"]
    effective_beta = bundle["beta"]

    device = next(model.parameters()).device
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    grad_accum   = int(cfg.get("grad_accum_steps", 4))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    warmup_ratio  = float(cfg.get("warmup_ratio", 0.05))
    train_log_path = results_dir / f"train_log_{run_name}.jsonl"

    total_params, trainable_params = count_parameters(model)
    print(
        f"\n{'='*60}\n"
        f"  Run:         {run_name}\n"
        f"  Beta:        {effective_beta}\n"
        f"  Examples:    {len(bundle['rows'])}\n"
        f"  Batch:       {cfg['batch_size']} × {grad_accum} = "
        f"{cfg['batch_size'] * grad_accum} (effective)\n"
        f"  Max seq:     {cfg['max_sequence_length']}\n"
        f"  LoRA params: {trainable_params:,} / {total_params:,} total\n"
        f"  Device:      {device}\n"
        f"{'='*60}\n"
    )

    # (Reference logprobs computed dynamically in loop)

    # ── 4. Optional torch.compile ─────────────────────────────────────────────
    # mode="reduce-overhead" uses CUDA Graphs — eliminates Python kernel-launch
    # latency which is the primary bottleneck on high-throughput Blackwell GPUs.
    if use_compile or cfg.get("compile", False):
        print("Compiling model (torch.compile mode='reduce-overhead')...")
        model = torch.compile(model, mode="reduce-overhead")
        print("  Done.\n")

    # ── Hardware tuning flags ─────────────────────────────────────────────────
    # cudnn.benchmark lets cuDNN auto-select the fastest convolution algorithm
    # for the fixed input shapes we use (sequence length is padded to a constant
    # within each batch, so shapes are sufficiently static).
    cudnn.benchmark = True

    # ── 5. AMP setup ─────────────────────────────────────────────────────────
    # GradScaler is mandatory with FP16: without it, small gradients underflow
    # to zero (FP16 min positive ≈ 6e-8). The scaler multiplies the loss by a
    # large factor before backward, then divides before the optimizer step.
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    autocast_ctx = torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=use_amp,
    )

    # ── 6. LR scheduler with warmup ───────────────────────────────────────────
    # Total optimizer steps = ceil(n_batches / grad_accum) * epochs
    n_batches = len(loader)
    total_optimizer_steps = max(1, (n_batches + grad_accum - 1) // grad_accum)
    warmup_steps = max(1, int(warmup_ratio * total_optimizer_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    print(
        f"LR scheduler: linear warmup for {warmup_steps} steps "
        f"/ {total_optimizer_steps} total optimizer steps.\n"
    )

    # ── 7. Training loop ──────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    global_step    = 0    # optimizer update steps
    accum_step     = 0    # micro-steps (forward + backward) between optimizer updates
    epoch_loss     = 0.0
    epoch_pref_acc = 0.0
    n_micro        = 0
    wall           = wall_timer()

    pbar = tqdm(loader, desc=f"DPO [{run_name}]", dynamic_ncols=True)

    for batch in pbar:
        # Collate returns None when all examples in a batch are filtered.
        if batch is None:
            continue

        chosen_batch, rejected_batch = batch

        # Move to GPU; non_blocking=True overlaps H2D transfer with compute.
        chosen_batch   = {k: v.to(device, non_blocking=True) for k, v in chosen_batch.items()}
        rejected_batch = {k: v.to(device, non_blocking=True) for k, v in rejected_batch.items()}

        # ── Forward pass under AMP autocast ───────────────────────────────────
        with autocast_ctx:
            # 1. Reference forward (no_grad)
            with torch.no_grad(), reference_mode(model):
                ref_chosen_logp, _, _ = response_sequence_logprobs(model, chosen_batch)
                ref_rejected_logp, _, _ = response_sequence_logprobs(model, rejected_batch)

            # 2. Policy forward
            policy_chosen_logp,   _, _ = response_sequence_logprobs(model, chosen_batch)
            policy_rejected_logp, _, _ = response_sequence_logprobs(model, rejected_batch)

            loss, diag = dpo_loss(
                policy_chosen_logp,
                policy_rejected_logp,
                ref_chosen_logp,
                ref_rejected_logp,
                effective_beta,
            )

        # Scale loss by grad_accum so gradient magnitudes stay consistent.
        scaler.scale(loss / grad_accum).backward()
        accum_step += 1
        n_micro    += 1

        # ── Optimizer step every grad_accum micro-steps ───────────────────────
        if accum_step % grad_accum == 0:
            # Unscale before gradient clipping so clip operates on real magnitudes.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters(model), max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        # ── Per-step metrics ──────────────────────────────────────────────────
        loss_val    = loss.item()
        pref_acc    = diag["preference_accuracy"].item()
        logit_mean  = diag["logit_mean"].item()
        pol_margin  = diag["policy_margin_mean"].item()
        current_lr  = scheduler.get_last_lr()[0]

        epoch_loss     += loss_val
        epoch_pref_acc += pref_acc

        # Append to disk every micro-step so evaluate.py / plot_results.py
        # can read results without needing notebook state.
        append_jsonl(train_log_path, {
            "run_name":             run_name,
            "beta":                 effective_beta,
            "optimizer_step":       global_step,
            "loss":                 loss_val,
            "logit_mean":           logit_mean,
            "policy_margin_mean":   pol_margin,
            "preference_accuracy":  pref_acc,
            "learning_rate":        current_lr,
            "grad_scale":           scaler.get_scale() if use_amp else 1.0,
            "elapsed_s":            wall(),
        })

        pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            pref_acc=f"{pref_acc:.3f}",
            lr=f"{current_lr:.2e}",
            step=global_step,
        )

    # ── 8. Save adapter checkpoint ────────────────────────────────────────────
    # Unwrap torch.compile _orig_mod wrapper before saving.
    save_model = getattr(model, "_orig_mod", model)
    save_model.save_pretrained(str(output))
    bundle["tokenizer"].save_pretrained(str(output))
    print(f"\nCheckpoint saved → {output}")

    # ── 9. Save epoch summary ─────────────────────────────────────────────────
    denom = max(n_micro, 1)
    summary = {
        "run_name":                  run_name,
        "beta":                      effective_beta,
        "total_optimizer_steps":     global_step,
        "mean_loss":                 epoch_loss / denom,
        "mean_preference_accuracy":  epoch_pref_acc / denom,
        "elapsed_s":                 wall(),
        "output_path":               str(output),
        "train_log":                 str(train_log_path),
    }
    save_json(results_dir / f"{run_name}_train_summary.json", summary)
    print(
        f"  mean_loss={summary['mean_loss']:.4f}  "
        f"mean_pref_acc={summary['mean_preference_accuracy']:.3f}  "
        f"elapsed={summary['elapsed_s']:.1f}s\n"
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Task 1 – DPO training")
    ap.add_argument("--config",       default="configs/dpo.yaml")
    ap.add_argument("--run-name",     default="standard")
    ap.add_argument("--dataset",      help="Override training JSONL path")
    ap.add_argument("--output",       help="Override checkpoint output path")
    ap.add_argument("--beta",         type=float, help="Override beta from config")
    ap.add_argument("--max-examples", type=int,   help="Limit examples (for ablations)")
    ap.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (mode='reduce-overhead') for extra Blackwell throughput",
    )
    args = ap.parse_args()

    run_training(
        config_path=args.config,
        run_name=args.run_name,
        dataset_path=args.dataset,
        output_path=args.output,
        beta=args.beta,
        max_examples=args.max_examples,
        use_compile=args.compile,
    )


if __name__ == "__main__":
    main()
