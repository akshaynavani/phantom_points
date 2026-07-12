"""Train the PointNet refiner on paired scans.

    python -m phantom_points.train --data out/pairs --epochs 20 --out runs/refiner

Loss = smooth-L1(range residual) + w_i * smooth-L1(intensity residual)   [matched points]
     + w_d * BCE(drop logit, unmatched)                                   [all real points]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import SectorPairDataset
from .export import export_weights
from .io import Dataset
from .model import PointNetRefiner, RefinerConfig


def refiner_loss(model: PointNetRefiner, batch, w_int: float = 1.0, w_drop: float = 0.5):
    raw = model(batch["feats"], batch["mask"])
    d_range, d_int, drop_logit = model.decode(raw)
    mask = batch["mask"]
    reg = batch["reg_mask"] & mask

    n_reg = reg.sum().clamp(min=1)
    l_range = (F.smooth_l1_loss(d_range, batch["d_range"], reduction="none", beta=0.05) * reg).sum() / n_reg
    l_int = (F.smooth_l1_loss(d_int, batch["d_int"], reduction="none", beta=0.02) * reg).sum() / n_reg
    bce = F.binary_cross_entropy_with_logits(drop_logit, batch["drop"], reduction="none")
    l_drop = (bce * mask).sum() / mask.sum().clamp(min=1)

    total = l_range + w_int * l_int + w_drop * l_drop
    with torch.no_grad():
        pred_drop = (drop_logit > 0).float()
        drop_acc = ((pred_drop == batch["drop"]).float() * mask).sum() / mask.sum().clamp(min=1)
        range_mae = ((d_range - batch["d_range"]).abs() * reg).sum() / n_reg
    return total, {
        "loss": total.item(),
        "range": l_range.item(),
        "intensity": l_int.item(),
        "drop": l_drop.item(),
        "drop_acc": drop_acc.item(),
        "range_mae": range_mae.item(),
    }


def _mean(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset written by phantom_gen --pair-config")
    ap.add_argument("--out", default="runs/refiner")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sectors", type=int, default=32)
    ap.add_argument("--max-points", type=int, default=4096)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ids = list(Dataset(args.data).ids)
    random.shuffle(ids)
    n_val = max(1, int(len(ids) * args.val_fraction))
    val_ids, train_ids = ids[:n_val], ids[n_val:] or ids[:n_val]

    common = dict(sectors=args.sectors, max_points=args.max_points)
    train_ds = SectorPairDataset(args.data, train_ids, seed=args.seed, **common)
    val_ds = SectorPairDataset(args.data, val_ids, seed=args.seed + 1, **common)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=len(train_ds) > args.batch_size)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    model = PointNetRefiner(RefinerConfig()).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_rows = []
        for batch in train_dl:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss, stats = refiner_loss(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_rows.append(stats)
        sched.step()

        model.eval()
        val_rows = []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(args.device) for k, v in batch.items()}
                val_rows.append(refiner_loss(model, batch)[1])

        tr, va = _mean(train_rows), _mean(val_rows)
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(f"epoch {epoch:3d}  train {tr.get('loss', 0):.4f}  val {va.get('loss', 0):.4f}  "
              f"range_mae {va.get('range_mae', 0) * 100:.2f} cm  drop_acc {va.get('drop_acc', 0):.3f}")
        torch.save(model.state_dict(), out / "last.pt")
        if va and va["loss"] < best:
            best = va["loss"]
            torch.save(model.state_dict(), out / "best.pt")

    (out / "history.json").write_text(json.dumps(history, indent=2))
    model.load_state_dict(torch.load(out / "best.pt", map_location=args.device))
    export_weights(model.cpu(), out / "refiner.ppnt", sectors=args.sectors)
    print(f"exported {out / 'refiner.ppnt'}")


if __name__ == "__main__":
    main()
