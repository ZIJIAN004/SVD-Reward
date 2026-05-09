import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from pathlib import Path

from config import Config
from tsp_data import generate_dataset
from model import TourAutoEncoder


def pos_weight_for(num_nodes: int) -> float:
    total = num_nodes * (num_nodes - 1) // 2     # all upper-triangle edges
    pos = num_nodes                               # tour edges
    return (total - pos) / pos


def train_epoch(model, loader, optim, pw, device):
    model.train()
    tot_loss, cnt = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        _, scores = model(batch)
        loss = F.binary_cross_entropy_with_logits(
            scores, batch.target,
            pos_weight=torch.tensor(pw, device=device),
        )
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        tot_loss += loss.item() * batch.num_graphs
        cnt += batch.num_graphs
    return tot_loss / cnt


@torch.no_grad()
def eval_epoch(model, loader, pw, device):
    model.eval()
    tot_loss, cnt, correct, total_edges = 0.0, 0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        _, scores = model(batch)
        loss = F.binary_cross_entropy_with_logits(
            scores, batch.target,
            pos_weight=torch.tensor(pw, device=device),
        )
        correct += ((scores > 0).float() == batch.target).sum().item()
        total_edges += batch.target.shape[0]
        tot_loss += loss.item() * batch.num_graphs
        cnt += batch.num_graphs
    return tot_loss / cnt, correct / total_edges


def train(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Generating training data …")
    train_data, _, train_good, train_iids = generate_dataset(
        cfg.num_train_instances, cfg.num_nodes,
        cfg.num_good_solutions, cfg.num_random_solutions,
        seed=cfg.seed,
    )
    print("Generating validation data …")
    val_data, _, _, _ = generate_dataset(
        cfg.num_val_instances, cfg.num_nodes,
        cfg.num_good_solutions, cfg.num_random_solutions,
        seed=cfg.seed + 1,
    )
    print(f"Train: {len(train_data)}  Val: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size)

    model = TourAutoEncoder(cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5)
    pw = pos_weight_for(cfg.num_nodes)
    print(f"pos_weight = {pw:.1f}")

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val, patience_cnt = float("inf"), 0
    for ep in range(1, cfg.num_epochs + 1):
        t_loss = train_epoch(model, train_loader, optim, pw, device)
        v_loss, v_acc = eval_epoch(model, val_loader, pw, device)
        sched.step(v_loss)
        lr_now = optim.param_groups[0]["lr"]
        print(f"Ep {ep:3d}  train_loss={t_loss:.4f}  "
              f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}  lr={lr_now:.1e}")

        if v_loss < best_val:
            best_val = v_loss
            patience_cnt = 0
            torch.save(
                {"model": model.state_dict(), "config": cfg.__dict__,
                 "epoch": ep, "val_loss": v_loss},
                save_dir / "best_model.pt",
            )
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                print(f"Early stopping at epoch {ep}")
                break

    print(f"Best val_loss = {best_val:.4f}")
    return model, train_data, train_good, train_iids


if __name__ == "__main__":
    train()
