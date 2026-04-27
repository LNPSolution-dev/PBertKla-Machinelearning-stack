"""
fold_*/predictions.npy 들을 평균내서 부모 디렉토리에 predictions.npy로 저장.
"""
import os, sys, glob
import numpy as np

if len(sys.argv) != 2:
    sys.exit("Usage: python avg_dl_preds.py <parent_dir_with_fold_subdirs>")

parent = sys.argv[1]
preds = []
for fd in sorted(glob.glob(os.path.join(parent, "fold_*"))):
    p = os.path.join(fd, "predictions.npy")
    if not os.path.exists(p):
        sys.exit(f"❌ 없음: {p}")
    a = np.load(p).ravel()
    preds.append(a)
    print(f"  loaded {p}: shape={a.shape}, mean={a.mean():.4f}")

if len(preds) != 5:
    print(f"⚠️  fold 수 {len(preds)}개 (5개 기대)")

avg = np.mean(preds, axis=0)
out = os.path.join(parent, "predictions.npy")
np.save(out, avg)
print(f"✅ saved {out}: shape={avg.shape}, mean={avg.mean():.4f}")
