"""
PBertKla + ML 스태킹 파이프라인 - 변경 전 vs 변경 후 종합 비교 보고서
- 내부 test (각 데이터셋의 자체 test CSV)
- 외부 16샘플 test (new_data_v1.csv)
- DL baseline (5-fold 평균)
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak,
)

# ── 한글 폰트 등록 ──
pdfmetrics.registerFont(TTFont("Nanum",     "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"))
pdfmetrics.registerFont(TTFont("NanumBold", "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"))

styles = getSampleStyleSheet()
H_TITLE = ParagraphStyle("Title", parent=styles["Title"],
                         fontName="NanumBold", fontSize=20, leading=26,
                         alignment=TA_CENTER, spaceAfter=10)
H_SUB   = ParagraphStyle("Sub", parent=styles["Normal"],
                         fontName="Nanum", fontSize=11, leading=15,
                         alignment=TA_CENTER,
                         textColor=colors.HexColor("#555"), spaceAfter=18)
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="NanumBold",
                    fontSize=15, leading=20, textColor=colors.HexColor("#1a3d6e"),
                    spaceBefore=14, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="NanumBold",
                    fontSize=12, leading=17, textColor=colors.HexColor("#2a5a9e"),
                    spaceBefore=8, spaceAfter=5)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Nanum",
                      fontSize=10, leading=15, alignment=TA_JUSTIFY, spaceAfter=5)
BULLET = ParagraphStyle("B", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=2)
NOTE = ParagraphStyle("N", parent=BODY, fontSize=9, textColor=colors.HexColor("#444"),
                      backColor=colors.HexColor("#f6f6f6"),
                      borderColor=colors.HexColor("#ddd"), borderWidth=0.5,
                      borderPadding=7, spaceBefore=3, spaceAfter=8)


def p(text, style=BODY): return Paragraph(text, style)
def h1(t): return Paragraph(t, H1)
def h2(t): return Paragraph(t, H2)
def bullets(items): return [Paragraph(f"• {t}", BULLET) for t in items]


def make_table(data, col_widths=None, header_bg="#1a3d6e",
               first_col_bold=True, font_size=8.5, body_align="CENTER"):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "NanumBold"),
        ("FONTNAME",   (0, 1), (-1, -1), "Nanum"),
        ("FONTSIZE",   (0, 0), (-1, -1), font_size),
        ("ALIGN",      (0, 0), (-1, -1), body_align),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#ccc")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f4f6fa")]),
    ]
    if first_col_bold:
        style.append(("FONTNAME", (0, 1), (0, -1), "NanumBold"))
        style.append(("ALIGN",    (0, 1), (0, -1), "LEFT"))
    t.setStyle(TableStyle(style))
    return t


# ============================================================
# 데이터 수집
# ============================================================
BASE = "/home/work/LNP_TEST/git_tools/PBertKla_v2"

# 내부 test - OLD (results/ML_output/data{N}/results_summary.json)
def load_old_internal(name):
    with open(f"{BASE}/results/ML_output/{name}/results_summary.json") as f:
        return json.load(f)

# 내부 test - NEW (results/ML_output/data{N}_oof/results_summary.json)
def load_new_internal(name):
    with open(f"{BASE}/results/ML_output/{name}_oof/results_summary.json") as f:
        return json.load(f)

# 외부 16샘플 - OLD (paper_figure/roc_curve/newdata_data{N}_ML/results_summary.json)
def load_old_external(name):
    path = f"{BASE}/paper_figure/roc_curve/newdata_{name}_ML/results_summary.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

# 외부 16샘플 - NEW (results/Inference_oof/data{N}_newdata/results_summary.json)
def load_new_external(name):
    with open(f"{BASE}/results/Inference_oof/{name}_newdata/results_summary.json") as f:
        return json.load(f)

# 16샘플에 대한 DL 5-fold 평균 + per-fold AUC
def dl_external_aucs(name):
    csv = f"{BASE}/Data/4_infer_new_data/new_data_v1.csv"
    y_true = pd.read_csv(csv)["label"].values
    folds = []
    for k in range(1, 6):
        p = np.load(f"{BASE}/results/Inference/DL_preds_5fold/{name}_newdata/fold_{k}/predictions.npy").ravel()
        folds.append(roc_auc_score(y_true, p))
    avg_pred = np.load(f"{BASE}/results/Inference/DL_preds_5fold/{name}_newdata/predictions.npy").ravel()
    avg_auc = roc_auc_score(y_true, avg_pred)
    avg_acc = accuracy_score(y_true, (avg_pred >= 0.5).astype(int))
    return folds, avg_auc, avg_acc


# DL 내부 test 5-fold 평균 AUC (ML 학습 로그에서 추출했던 값)
DL_INTERNAL_AUC = {
    "data1": 0.8827,
    "data2": 0.8881,
    "data3": 0.9063,
}

OLD_INT = {n: load_old_internal(n) for n in ["data1", "data2", "data3"]}
NEW_INT = {n: load_new_internal(n) for n in ["data1", "data2", "data3"]}
OLD_EXT = {n: load_old_external(n) for n in ["data1", "data2", "data3"]}
NEW_EXT = {n: load_new_external(n) for n in ["data1", "data2", "data3"]}
DL_EXT  = {n: dl_external_aucs(n) for n in ["data1", "data2", "data3"]}

# ============================================================
# 보고서 생성
# ============================================================
story = []

# === 표지 ===
story.append(Spacer(1, 2.5*cm))
story.append(p("PBertKla + ML 스태킹 파이프라인", H_TITLE))
story.append(p("변경 전 vs 변경 후 종합 비교 보고서", H_TITLE))
story.append(Spacer(1, 0.4*cm))
story.append(p("내부 test, 외부 16샘플 test, DL baseline 모두 포함", H_SUB))
story.append(Spacer(1, 3.5*cm))

meta = [
    ["문서 정보", ""],
    ["대상 파이프라인", "PBertKla 5-fold 파인튜닝 + LightGBM/XGBoost/CatBoost 스태킹"],
    ["변경 사항", "ML train 메타 피처: LightGBM-OOF → Transformer-OOF (정석 stacking)"],
    ["내부 test", "각 데이터셋 자체 test split (data1: 1,928 / data2: 2,595 / data3: 5,207)"],
    ["외부 OOD test", "Data/4_infer_new_data/new_data_v1.csv (16 samples, 양성 8 + 음성 8)"],
    ["변경 전 결과 출처", "results/ML_output/data{N}/, paper_figure/roc_curve/newdata_data{N}_ML/"],
    ["변경 후 결과 출처", "results/ML_output/data{N}_oof/, results/Inference_oof/data{N}_newdata/"],
    ["작성일", "2026-04-26"],
]
story.append(make_table(meta, col_widths=[4.2*cm, 11*cm], font_size=9.5))
story.append(PageBreak())

# === 1. 요약 ===
story.append(h1("1. 요약 (TL;DR)"))
story.append(p(
    "기존 파이프라인에서 ML 학습 시점의 메타 피처를 LightGBM OOF에서 "
    "Transformer 5-fold OOF로 교체하고, ML 추론 시점의 메타도 단일 fold에서 "
    "Transformer 5-fold 평균으로 통일하여 정석 Stacked Generalization을 구현했다. "
    "본 보고서는 이 변경의 실제 효과를 두 가지 평가셋에서 정량 비교한다."
))
story.append(Spacer(1, 4))

summary_table = [
    ["평가셋", "변경 전", "변경 후", "변화"],
    ["내부 test (n=1928~5207)",
     "ML AUC 0.62~0.91 (모델별 편차 큼)",
     "ML AUC 0.88~0.91 (DL 수준 또는 그 이상으로 수렴)",
     "✅ 모든 데이터셋에서 개선"],
    ["외부 16샘플 OOD",
     "ML AUC 0.45~0.72 (cherry-picked 결과)",
     "ML AUC 0.36~0.52 (정직한 결과)",
     "⚠️ 수치 하락 — 이유는 본문 5장"],
]
story.append(make_table(summary_table, col_widths=[3.8*cm, 4.8*cm, 4.8*cm, 2.4*cm],
                        font_size=8.5))
story.append(Spacer(1, 4))
story.append(p(
    "<b>핵심 메시지</b>: 내부 test에서는 변경 후 모델이 모든 면에서 개선됐다. "
    "외부 16샘플에서는 수치가 하락한 것처럼 보이지만, 이는 변경 전 결과가 fold 선택과 "
    "분포 mismatch가 만든 over-optimistic 수치였음이 드러난 것이다. "
    "16샘플 자체가 통계적으로 너무 작고, transformer가 이 OOD 분포에 일반화하지 못하는 "
    "근본적 한계가 정직하게 드러났다.",
    NOTE
))

# === 2. 변경 사항 한눈에 ===
story.append(h1("2. 변경 사항 한눈에"))
change_table = [
    ["항목", "변경 전 (Before)", "변경 후 (After)"],
    ["ML 학습 train 메타", "LightGBM OOF (sequence features)", "Transformer 5-fold OOF"],
    ["ML 추론 test 메타", "Transformer 단일 fold (e.g. fold_2)", "Transformer 5-fold 평균"],
    ["Train·Test 메타 분포", "❌ 다름 (LGBM vs Transformer)", "✅ 동일 (Transformer · Transformer)"],
    ["DL 5-fold 활용", "1개 fold만 사용", "5개 fold 모두 사용 (OOF + avg)"],
    ["방법론 분류", "비표준 변형", "표준 Stacked Generalization"],
    ["DL 재학습 필요?", "—", "❌ 기존 가중치 재활용 (OOF 생성만 추가)"],
]
story.append(make_table(change_table, col_widths=[4.0*cm, 5.5*cm, 6.0*cm], font_size=8.5))

# === 3. 내부 test 결과 비교 ===
story.append(PageBreak())
story.append(h1("3. 내부 test 결과 (각 데이터셋의 자체 test CSV)"))
story.append(p(
    "각 데이터셋의 train CSV로 학습하고, 그 데이터셋의 test CSV로 평가한 결과. "
    "test 샘플 수가 충분히 크므로 수치 신뢰도가 높다."
))

for ds, n_test in [("data1", 1928), ("data2", 2595), ("data3", 5207)]:
    story.append(h2(f"3.{['data1','data2','data3'].index(ds)+1} {ds}  (n_test = {n_test})"))
    rows = [["모델", "변경 전 AUC", "변경 후 AUC", "변경 전 F1", "변경 후 F1"]]
    for m, key in [("LightGBM", "lgbm"), ("XGBoost", "xgb"),
                   ("CatBoost", "catboost"), ("Ensemble", "ensemble")]:
        old = OLD_INT[ds][key]
        new = NEW_INT[ds][key]
        rows.append([
            m,
            f"{old['auc_roc']:.4f}",
            f"{new['auc_roc']:.4f}",
            f"{old['f1']:.4f}",
            f"{new['f1']:.4f}",
        ])
    rows.append(["— DL 5-fold avg —",
                 "—", f"{DL_INTERNAL_AUC[ds]:.4f}", "—", "—"])
    story.append(make_table(rows,
                            col_widths=[3.2*cm, 2.8*cm, 2.8*cm, 2.6*cm, 2.6*cm],
                            font_size=8.5))
    story.append(Spacer(1, 4))

story.append(p(
    "<b>관찰</b>: 변경 후 모든 ML 모델이 DL 5-fold 평균 AUC와 비슷하거나 "
    "더 높은 수준으로 수렴한다. 특히 data1에서는 LGBM/XGB가 변경 전 0.62~0.68 → "
    "변경 후 0.88로 큰 폭 개선됐다. 이는 train·test 메타 분포 일치로 ML이 학습한 "
    "결합 규칙이 비로소 test에서 제대로 일반화됨을 의미한다.",
    NOTE
))

# === 4. 외부 16샘플 OOD test 결과 ===
story.append(PageBreak())
story.append(h1("4. 외부 16샘플 OOD test (Data/4_infer_new_data/new_data_v1.csv)"))
story.append(p(
    "양성 8개 (ALDOA_230, ALDOA2_322, cGAS_131, TEAD1_108, YAP_90, CPT1_457, "
    "CPT2_458, PDHA1_336) + 음성 8개 (NonKla1~8). 학습 데이터와 분포가 다른 외부 셋."
))

for ds in ["data1", "data2", "data3"]:
    story.append(h2(f"4.{['data1','data2','data3'].index(ds)+1} {ds}"))

    folds, avg_auc, avg_acc = DL_EXT[ds]
    fold_str = " / ".join([f"{a:.3f}" for a in folds])
    story.append(p(
        f"<b>DL per-fold AUC</b>: {fold_str}<br/>"
        f"<b>DL 5-fold avg AUC</b>: {avg_auc:.3f}  (acc {avg_acc:.3f})",
        NOTE
    ))

    rows = [["모델", "변경 전 AUC", "변경 후 AUC", "변경 전 F1 / Acc", "변경 후 F1 / Acc"]]

    if OLD_EXT[ds] is not None:
        old_map = {"LightGBM": "LightGBM", "XGBoost": "XGBoost",
                   "CatBoost": "CatBoost", "Ensemble": "Ensemble (Option A)"}
        new_map = {"LightGBM": "lgbm", "XGBoost": "xgb",
                   "CatBoost": "catboost", "Ensemble": "ensemble"}
        for m in ["LightGBM", "XGBoost", "CatBoost", "Ensemble"]:
            old = OLD_EXT[ds][old_map[m]]
            new = NEW_EXT[ds][new_map[m]]
            rows.append([
                m,
                f"{old['roc_auc']:.4f}",
                f"{new['auc_roc']:.4f}",
                f"— / {old['acc']:.3f}",
                f"{new['f1']:.3f} / {new['accuracy']:.3f}",
            ])
    else:
        # data2 — 이전 16샘플 결과 없음
        for m, key in [("LightGBM","lgbm"), ("XGBoost","xgb"),
                       ("CatBoost","catboost"), ("Ensemble","ensemble")]:
            new = NEW_EXT[ds][key]
            rows.append([
                m, "—", f"{new['auc_roc']:.4f}",
                "—", f"{new['f1']:.3f} / {new['accuracy']:.3f}"
            ])
        story.append(p("⚠️ data2는 변경 전 외부 16샘플 결과가 없습니다.", NOTE))

    story.append(make_table(rows,
                            col_widths=[2.8*cm, 2.6*cm, 2.6*cm, 3.2*cm, 3.2*cm],
                            font_size=8.5))
    story.append(Spacer(1, 6))

# === 5. 외부 16샘플에서 수치가 낮아진 이유 ===
story.append(PageBreak())
story.append(h1("5. 외부 16샘플에서 수치가 낮아진 이유 (분석)"))

story.append(h2("5.1 Fold cherry-picking이 사라짐"))
story.append(p(
    "변경 전 파이프라인은 5개 fold 중 한 개(예: data3 fold_2)만 inference에 사용했다. "
    "16샘플 OOD test에 대한 per-fold AUC를 보면:"
))
fold_compare = [
    ["데이터셋", "fold 1", "fold 2", "fold 3", "fold 4", "fold 5", "5-fold 평균"],
]
for ds in ["data1", "data2", "data3"]:
    folds, avg, _ = DL_EXT[ds]
    fold_compare.append([ds] + [f"{a:.3f}" for a in folds] + [f"{avg:.3f}"])
story.append(make_table(fold_compare,
                        col_widths=[1.8*cm] + [1.7*cm]*5 + [2.4*cm],
                        font_size=8.5))
story.append(Spacer(1, 4))
story.append(p(
    "<b>data3에서 fold_2 = 0.578이 가장 좋은 fold 중 하나</b>였다. 변경 전 추론은 우연히 "
    "이 lucky fold를 선택했고, 변경 후 5-fold 평균(0.406)은 다른 약한 fold들과 섞여 "
    "낮아진다. 또한 16개라는 작은 표본에서 fold 간 예측이 다른 샘플에서 엇갈려 "
    "<b>평균 자체의 AUC가 per-fold AUC보다 낮아지는 현상</b>도 발생했다 "
    "(data3: 평균 0.406 < per-fold 최댓값 0.594).",
    NOTE
))

story.append(h2("5.2 Transformer가 이 OOD set에 일반화하지 않음"))
story.append(p(
    "data1 transformer는 모든 5개 fold에서 AUC < 0.5로 사실상 분류 능력이 없다. "
    "data2도 5개 중 1개만 0.5를 넘는다. 이는 학습 데이터의 분포와 16샘플 new_data_v1의 "
    "분포가 매우 다르다는 강력한 신호다. 변경 후 ML 모델은 이 OOD-에 약한 transformer "
    "신호를 train 메타로 받아 학습했으므로, ML도 OOD에서 약해진다."
))

story.append(h2("5.3 변경 전이 우연히 OOD에서 더 잘 보였던 이유"))
story.append(p(
    "변경 전 파이프라인은 train 메타로 LightGBM(sequence-only) OOF를 사용했다. "
    "ML 모델은 학습 시점에 \"transformer 분포\"가 아닌 \"sequence 통계로부터 도출된 "
    "확률 분포\"를 보고 학습했다. 그런데 OOD인 16샘플에서는 transformer가 제 역할을 못 "
    "하므로, ML이 transformer 신호에 덜 의존하는 변경 전 구조가 우연히 더 robust해 보인 "
    "것이다. 즉, 변경 전 0.719라는 수치는 \"올바른 stacking이 아니어서\" 우연히 "
    "더 좋게 나온 케이스다."
))
story.extend(bullets([
    "변경 전 data1 LightGBM AUC 0.719: ML이 transformer를 학습 시 안 봐서 OOD에서도 "
    "transformer의 잘못된 신호에 끌려가지 않음 (happy accident).",
    "변경 후 data1 LightGBM AUC 0.453: ML이 transformer를 정직하게 메타로 받음 → "
    "transformer가 잘못된 OOD에서 함께 잘못됨.",
    "어느 쪽이 \"진짜\" 모델 능력인가? <b>변경 후가 정확한 측정</b>이고, "
    "변경 전이 우연이 만든 over-estimate.",
]))

story.append(h2("5.4 16샘플은 통계적으로 너무 작음"))
story.append(p(
    "양성 8 + 음성 8개의 16샘플 평가는 한 샘플 차이가 AUC 1/64 = 0.0156, accuracy "
    "1/16 = 6.25%를 움직인다. 두 모델 비교에 의미를 부여하려면 차이가 적어도 0.05 이상 "
    "나야 하는데, 본 보고서의 거의 모든 비교는 그 이하 또는 그 경계에 있다. "
    "어떤 결과도 \"통계적으로 유의\"하다고 주장할 수 없다."
))

# === 6. 결론 및 권고 ===
story.append(PageBreak())
story.append(h1("6. 결론 및 권고"))

story.append(h2("6.1 본 변경의 평가"))
story.append(p(
    "정석 Stacked Generalization 적용은 다음 두 가지 효과를 모두 달성했다:"
))
story.extend(bullets([
    "<b>내부 test에서 명백한 개선</b>: ML 모델들이 DL baseline 수준 또는 그 이상으로 "
    "수렴 (data1 LightGBM 0.62→0.88). 이는 train·test 메타 분포 일치로 ML이 학습한 "
    "결합 규칙이 비로소 test에서도 의미를 가지게 된 결과.",
    "<b>외부 16샘플에서 정직한 성능 노출</b>: 이전의 0.72같은 수치는 fold cherry-picking "
    "+ 분포 mismatch가 만든 over-estimate. 변경 후 수치가 모델의 진짜 OOD 일반화 능력 "
    "(즉, 현재로선 부족함) 을 보여줌. 이는 결함이 아니라 진단 정보."
]))

story.append(h2("6.2 논문 보고를 위한 권고"))
story.extend(bullets([
    "<b>주요 결과는 내부 test로 보고</b>. data1/2/3 모두 AUC 0.88~0.91 수준으로 "
    "충분히 강한 결과. 변경 전 vs 후 비교를 ablation으로 함께 표시.",
    "<b>외부 16샘플 결과는 \"OOD evaluation\"으로 명시적 라벨링</b>. 수치가 낮음을 "
    "honest하게 보고하고, 학습 데이터와 분포 차이가 원인임을 설명.",
    "<b>외부 평가셋 확장 강력 권고</b>. 16샘플은 통계적 유의성을 주장할 수 없는 크기. "
    "수백 샘플 이상의 외부 검증셋(다른 단백질 family, 다른 PTM database)을 확보해야 paper "
    "review에서 신뢰성 있는 일반화 능력 주장 가능.",
    "<b>다음 개선 방향</b>: sequence feature를 transformer와 직교한 것 (ESM/ProtT5 "
    "embedding, conservation, structure) 으로 교체하면 ML이 transformer가 놓치는 신호를 "
    "보충해 OOD에서도 robust해질 가능성.",
]))

# 빌드
out_path = f"{BASE}/PBertKla_pipeline_comparison_report.pdf"
doc = SimpleDocTemplate(
    out_path, pagesize=A4,
    leftMargin=1.8*cm, rightMargin=1.8*cm,
    topMargin=1.8*cm, bottomMargin=1.8*cm,
    title="PBertKla Pipeline Comparison Report",
    author="lnpsolution@lnpsolution.com",
)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Nanum", 8)
    canvas.setFillColor(colors.HexColor("#888"))
    canvas.drawString(1.8*cm, 1.0*cm, "PBertKla 변경 전 vs 변경 후 종합 비교 보고서")
    canvas.drawRightString(A4[0]-1.8*cm, 1.0*cm, f"- {doc.page} -")
    canvas.restoreState()


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"✅ PDF 생성 완료: {out_path}")
print(f"   파일 크기: {os.path.getsize(out_path)/1024:.1f} KB")
