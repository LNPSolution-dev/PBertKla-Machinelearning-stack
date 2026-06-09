#!/usr/bin/env python3
"""
R2-6: retrain PCBert-Kla's method on OUR data3 (Multi) and evaluate on OUR data3
test (n=5207), same blind test set as PBertKla-Stack.

Faithful to PCBert-Kla (Zhang et al. 2025, Brief. Bioinform.): ProtBert (Rostlab/
prot_bert) truncated to the first 4 transformer layers, CLS embedding fused with
27 BioPython physicochemical features, small attention+FC head. Their published
code is a single-commit notebook with a forward-pass bug (uses a global `inputs`
instead of the batch); we fixed that and wired it to our fixed train/test split.
"""
import os, json, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             f1_score, matthews_corrcoef, precision_score, recall_score)
from Bio.SeqUtils.ProtParam import ProteinAnalysis

os.environ.setdefault("HF_HOME", "/home/work/LNP_TEST/git_tools/PBertKla_v2/benchmark/.hf_cache")
STD = set("ACDEFGHIKLMNPQRSTVWY")


def physico(seq):
    """27-dim physicochemical features (PCBert-Kla, BioPython ProteinAnalysis)."""
    s = "".join(c for c in seq.upper() if c in STD)      # drop X/padding & non-standard
    if len(s) < 3:
        return [0.0] * 27
    pa = ProteinAnalysis(s)
    feats = [pa.molecular_weight(), pa.isoelectric_point()]
    feats += list(pa.amino_acids_percent.values())            # 20
    feats += list(pa.secondary_structure_fraction())          # 3 (helix, turn, sheet)
    feats += [pa.gravy(), pa.charge_at_pH(7.0)]
    return feats


class Attention(nn.Module):
    def __init__(self, d):
        super().__init__(); self.W = nn.Linear(d, 1); self.sig = nn.Sigmoid()
    def forward(self, x):
        return self.sig(self.W(x)) * x


class PCBertKla(nn.Module):
    def __init__(self, bert, n_feat=27):
        super().__init__()
        self.bert = bert
        self.fc1 = nn.Linear(bert.config.hidden_size + n_feat, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)
        self.att1 = Attention(32)
        self.relu = nn.ReLU(); self.sig = nn.Sigmoid()
        self.drop1 = nn.Dropout(0.1); self.drop2 = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, fe):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state[:, 0, :]          # CLS
        x = torch.cat((x, fe), dim=1)
        x = self.drop1(x)
        x = self.relu(self.fc1(x)); x = self.att1(x); x = self.drop2(x)
        x = self.relu(self.fc2(x)); x = self.drop2(x)
        return self.sig(self.fc3(x)).squeeze(-1)


def tok_seqs(seqs, tokenizer, max_len=64):
    spaced = [" ".join("".join(c if c in STD else "X" for c in s.upper())) for s in seqs]
    enc = tokenizer(spaced, padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def run(loader, model, device, train=False, opt=None, lossfn=None):
    model.train() if train else model.eval()
    ps, ys = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for ids, am, fe, y in loader:
            ids, am, fe, y = ids.to(device), am.to(device), fe.to(device), y.to(device)
            p = model(ids, am, fe)
            if train:
                loss = lossfn(p, y); opt.zero_grad(); loss.backward(); opt.step()
            ps.append(p.detach().cpu().numpy()); ys.append(y.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data3_train.csv")
    ap.add_argument("--test", default="data3_test.csv")
    ap.add_argument("--out", default="pcbert_data3")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tr = pd.read_csv(args.train); te = pd.read_csv(args.test)
    # our data3_*.csv from earlier are space-tokenized; rebuild raw seq
    def raw(df): return df["sequence"].str.replace(" ", "", regex=False)
    tr_seq, te_seq = raw(tr).tolist(), raw(te).tolist()
    ytr, yte = tr["label"].astype(np.float32).values, te["label"].astype(np.float32).values
    print(f"train={len(tr_seq)} test={len(te_seq)}", flush=True)

    t0 = time.time()
    Ftr = np.array([physico(s) for s in tr_seq], dtype=np.float32)
    Fte = np.array([physico(s) for s in te_seq], dtype=np.float32)
    sc = MinMaxScaler().fit(Ftr)
    Ftr, Fte = sc.transform(Ftr).astype(np.float32), sc.transform(Fte).astype(np.float32)
    print(f"features done ({time.time()-t0:.0f}s)", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
    bert = AutoModel.from_pretrained("Rostlab/prot_bert")
    del bert.encoder.layer[4:]                       # PCBert-Kla: keep first 4 layers
    model = PCBertKla(bert).to(device)

    ids_tr, am_tr = tok_seqs(tr_seq, tokenizer)
    ids_te, am_te = tok_seqs(te_seq, tokenizer)

    # 10% validation split for early stopping
    idx = np.arange(len(tr_seq))
    tri, vai = train_test_split(idx, test_size=0.1, random_state=args.seed, stratify=ytr)
    def ds(i, ids, am, F, y):
        return TensorDataset(ids[i], am[i], torch.tensor(F[i]), torch.tensor(y[i]))
    dl_tr = DataLoader(ds(tri, ids_tr, am_tr, Ftr, ytr), batch_size=args.bs, shuffle=True)
    dl_va = DataLoader(ds(vai, ids_tr, am_tr, Ftr, ytr), batch_size=128)
    dl_te = DataLoader(TensorDataset(ids_te, am_te, torch.tensor(Fte), torch.tensor(yte)),
                       batch_size=128)

    opt = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": 2e-5},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("bert.")], "lr": 1e-3},
    ])
    lossfn = nn.BCELoss()

    best_auc, best_state, bad = -1, None, 0
    for ep in range(1, args.epochs + 1):
        run(dl_tr, model, device, train=True, opt=opt, lossfn=lossfn)
        yv, pv = run(dl_va, model, device)
        va = roc_auc_score(yv, pv)
        print(f"epoch {ep}: val AUROC {va:.4f}", flush=True)
        if va > best_auc:
            best_auc, best_state, bad = va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ {ep}", flush=True); break

    model.load_state_dict(best_state)
    y, p = run(dl_te, model, device)
    pred = (p >= 0.5).astype(int)
    m = {"tool": "PCBert-Kla (retrained on data3)", "n_test": int(len(y)),
         "AUROC": float(roc_auc_score(y, p)), "AUPRC": float(average_precision_score(y, p)),
         "ACC": float(accuracy_score(y, pred)), "Precision": float(precision_score(y, pred)),
         "Recall": float(recall_score(y, pred)), "F1": float(f1_score(y, pred)),
         "MCC": float(matthews_corrcoef(y, pred)), "val_AUROC": float(best_auc),
         "minutes": round((time.time() - t0) / 60, 2)}
    np.save(args.out + "_proba.npy", p)
    json.dump(m, open(args.out + "_metrics.json", "w"), indent=2)
    print(json.dumps(m, indent=2), flush=True)


if __name__ == "__main__":
    main()
