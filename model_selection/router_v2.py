"""Leakage-free replacement for `meta_learning.py`.

Three changes from the original:

1. **No ground-truth feature.** The old feature vector began with a one-hot of
   `true_corruption` and the label was derived from a variable that copied it
   90% of the time, so the network could score ~90% by reading three inputs.

2. **One featuriser, shared with inference.** `agentnet_lc.router.featurise` is
   imported here rather than reimplemented, which is what stops the train and
   inference vectors from drifting apart the way they did before.

3. **LayerNorm, not BatchNorm.** Inference runs one case at a time; LayerNorm
   behaves identically in train and eval mode, removing a class of batch-size-
   dependent bugs.

Train on real logged cases, not synthetic ones:

    python -m agentnet_lc.cli --input data/corrupted/ --jsonl traces/cases.jsonl
    python model_selection/router_v2.py --traces traces/cases.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentnet_lc.router import FEATURE_DIM, ROUTE_ORDER, featurise  # noqa: E402
from agentnet_lc.schemas import AgentOpinion, CorruptionType  # noqa: E402


class RouterV2(nn.Module):
    def __init__(self, d_in: int = FEATURE_DIM, d_h: int = 64, d_out: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.LayerNorm(d_h),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_h, d_h // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_h // 2, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _opinions_from_row(row: dict) -> list[AgentOpinion]:
    out = []
    for o in row.get("opinions", []):
        label = o.get("classification")
        out.append(
            AgentOpinion(
                agent=o["agent"],
                classification=CorruptionType(label) if label else None,
                confidence=float(o.get("confidence") or 0.5),
                reasoning=o.get("reasoning", ""),
                ok=bool(o.get("ok", True)),
            )
        )
    return out


def load_traces(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (X, y) from logged cases.

    The label is `best_route` -- which restoration model actually produced the
    highest SSIM -- not a deterministic mapping from a classification. That is
    what turns the router from a label-copier into something with a reason to
    learn.
    """
    features, labels = [], []
    skipped = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        best = row.get("best_route") or (row.get("route") or {}).get("model")
        if best is None:
            skipped += 1
            continue
        try:
            target = [r.value for r in ROUTE_ORDER].index(best)
        except ValueError:
            skipped += 1
            continue
        features.append(featurise(_opinions_from_row(row)))
        labels.append(target)

    if not features:
        raise SystemExit(f"no usable rows in {path} ({skipped} skipped)")
    print(f"loaded {len(features)} cases ({skipped} skipped)")
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def train(traces: Path, out: Path, epochs: int = 60, seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    X, y = load_traces(traces)

    # Split before fitting anything. With real data, split by patient/study id
    # rather than by row -- adjacent slices are near-duplicates.
    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    cut = int(0.8 * len(X))
    tr, va = order[:cut], order[cut:]

    model = RouterV2()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        optimiser.zero_grad()
        loss = criterion(model(X[tr]), y[tr])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            acc = (model(X[va]).argmax(-1) == y[va]).float().mean().item() if len(va) else 0.0
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out)
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1:3d}  loss {loss.item():.4f}  val acc {acc:.3f}")

    print(f"best validation accuracy {best_acc:.3f}; saved {out}")
    print(
        "Compare this against the majority-vote and best-single-agent baselines "
        "before claiming the router adds value."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, default=Path("traces/cases.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("model_selection/router_v2.pth"))
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    train(args.traces, args.out, args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
