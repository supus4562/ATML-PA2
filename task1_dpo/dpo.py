from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    The correct DPO objective (Rafailov et al. 2023, Eq. 7):

        loss = -E[ log σ( β · [ (log π_θ(y⁺|x) - log π_θ(y⁻|x))
                                - (log π_ref(y⁺|x) - log π_ref(y⁻|x)) ] ) ]

    The key insight: the logits are policy_margin MINUS reference_margin.
    A higher β enforces a tighter KL constraint to the reference policy.

    NOTE: The starter code contained a deliberate bug where the reference
    margin was ADDED instead of subtracted — this has been corrected below.
    Validated against validate_dpo_loss() below.
    """

    # Policy model preference margin: how much more the policy prefers
    # the chosen response over the rejected one (in log-prob space).
    policy_margin = (
        policy_chosen_logp
        - policy_rejected_logp
    )

    # Reference policy preference margin: same quantity for the frozen
    # reference model (computed via model.disable_adapter() in train.py).
    ref_margin = (
        ref_chosen_logp
        - ref_rejected_logp
    )

    # BUG FIX: was `policy_margin + ref_margin` (WRONG).
    # Correct formula subtracts the reference margin so the loss penalises
    # the policy whenever it fails to improve over the reference on the
    # preference gap — not whenever it merely agrees with it.
    logits = beta * (
        policy_margin - ref_margin  # <-- FIXED: was `+`, now `-`
    )

    # Negative log-sigmoid of the logits (binary cross-entropy on the
    # implicit Bradley-Terry preference label).
    loss = -F.logsigmoid(logits).mean()

    return loss, {
        # Mean of the scalar DPO logits — should start near 0 and grow
        # positive as the policy diverges from the reference in the right dir.
        "logit_mean": logits.detach().mean(),

        # How much the policy (alone) prefers chosen over rejected.
        "policy_margin_mean": policy_margin.detach().mean(),

        # Fraction of examples where the policy improves its margin vs
        # the reference (the correct DPO preference accuracy definition).
        "preference_accuracy": (
            (policy_margin - ref_margin) > 0
        ).float().mean().detach(),
    }


def validate_dpo_loss():
    """Numeric sanity-check for dpo_loss.

    Called automatically by train.py before training starts.
    Uses synthetic tensors where the correct loss is known analytically.

    Test 1 – Perfect policy:
        If policy perfectly agrees with preferences and reference is neutral
        (zero margin), logits = beta * policy_margin > 0, loss < log(2).

    Test 2 – Bug-detection:
        With the old buggy `+` formula, a policy that matches the reference
        exactly would produce non-zero logits. With the correct `-` formula,
        identical policy and reference => logits == 0 => loss == log(2).

    Test 3 – Preference accuracy:
        When policy_margin > ref_margin for all examples, accuracy == 1.0.
    """
    device = torch.device("cpu")
    beta = 0.1

    # --- Test 1: Non-trivial loss value ---
    pc = torch.tensor([-1.0, -2.0], device=device)   # policy chosen logp
    pr = torch.tensor([-3.0, -5.0], device=device)   # policy rejected logp
    rc = torch.tensor([-2.0, -2.5], device=device)   # ref chosen logp
    rr = torch.tensor([-2.0, -2.5], device=device)   # ref rejected logp (same as rc → ref margin=0)

    loss, diag = dpo_loss(pc, pr, rc, rr, beta)

    # When ref_margin == 0, logits = beta * policy_margin
    policy_margin = pc - pr           # [2.0, 3.0]
    expected_logits = beta * policy_margin  # [0.2, 0.3]
    expected_loss = -F.logsigmoid(expected_logits).mean()
    assert abs(loss.item() - expected_loss.item()) < 1e-5, (
        f"Test 1 failed: got {loss.item():.6f}, expected {expected_loss.item():.6f}"
    )

    # --- Test 2: Policy == reference → logits must be 0 → loss == log(2) ---
    pc2 = torch.tensor([-1.0, -2.0], device=device)
    pr2 = torch.tensor([-3.0, -5.0], device=device)
    # ref has identical margin as policy
    rc2 = torch.tensor([-1.0, -2.0], device=device)
    rr2 = torch.tensor([-3.0, -5.0], device=device)

    loss2, _ = dpo_loss(pc2, pr2, rc2, rr2, beta)
    expected_loss2 = torch.log(torch.tensor(2.0))  # -log σ(0) = log(2)
    assert abs(loss2.item() - expected_loss2.item()) < 1e-5, (
        f"Test 2 (bug-detection) failed: got {loss2.item():.6f}, expected {expected_loss2.item():.6f}. "
        "If you see this, the `+` bug was NOT fixed — check the logits line."
    )

    # --- Test 3: Preference accuracy ---
    _, diag3 = dpo_loss(pc, pr, rc, rr, beta)
    # policy_margin=[2,3], ref_margin=[0,0] → all positive → accuracy=1.0
    assert abs(diag3["preference_accuracy"].item() - 1.0) < 1e-5, (
        f"Test 3 failed: preference_accuracy={diag3['preference_accuracy'].item()}"
    )

    print("[dpo_loss] All 3 validation checks passed. Bug fix confirmed.")
