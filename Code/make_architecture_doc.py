"""
PBertKla + ML 스태킹 모델 아키텍처 설명 PDF (≤5 페이지)
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# matplotlib 한글 폰트
fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
mpl.rcParams["font.family"] = "NanumGothic"
mpl.rcParams["axes.unicode_minus"] = False

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image,
)

# ── 한글 폰트 ──
pdfmetrics.registerFont(TTFont("Nanum",     "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"))
pdfmetrics.registerFont(TTFont("NanumBold", "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"))

styles = getSampleStyleSheet()
H_TITLE = ParagraphStyle("Title", parent=styles["Title"], fontName="NanumBold",
                         fontSize=18, leading=24, alignment=TA_CENTER, spaceAfter=8)
H_SUB   = ParagraphStyle("Sub", parent=styles["Normal"], fontName="Nanum",
                         fontSize=10.5, leading=14, alignment=TA_CENTER,
                         textColor=colors.HexColor("#555"), spaceAfter=14)
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="NanumBold",
                    fontSize=14, leading=18, textColor=colors.HexColor("#1a3d6e"),
                    spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="NanumBold",
                    fontSize=11.5, leading=15, textColor=colors.HexColor("#2a5a9e"),
                    spaceBefore=6, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Nanum",
                      fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=4)
BULLET = ParagraphStyle("B", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=2)
NOTE = ParagraphStyle("N", parent=BODY, fontSize=9, textColor=colors.HexColor("#444"),
                      backColor=colors.HexColor("#f6f6f6"),
                      borderColor=colors.HexColor("#ddd"), borderWidth=0.5,
                      borderPadding=6, spaceBefore=2, spaceAfter=6)
CAPTION = ParagraphStyle("C", parent=BODY, fontSize=8.5, alignment=TA_CENTER,
                         textColor=colors.HexColor("#666"), spaceBefore=2, spaceAfter=8)


def p(text, style=BODY): return Paragraph(text, style)
def h1(t): return Paragraph(t, H1)
def h2(t): return Paragraph(t, H2)
def bullets(items): return [Paragraph(f"• {t}", BULLET) for t in items]


def make_table(data, col_widths=None, header_bg="#1a3d6e", first_col_bold=True,
               font_size=8.5, body_align="CENTER"):
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
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f4f6fa")]),
    ]
    if first_col_bold:
        style.append(("FONTNAME", (0, 1), (0, -1), "NanumBold"))
        style.append(("ALIGN",    (0, 1), (0, -1), "LEFT"))
    t.setStyle(TableStyle(style))
    return t


# ============================================================
# 아키텍처 다이어그램 (matplotlib)
# ============================================================
def draw_architecture(out_path):
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor="white")
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

    def box(x, y, w, h, text, color, fc="white", fontsize=8.5, bold=False):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.4",
                              linewidth=1.4, edgecolor=color, facecolor=fc)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold" if bold else "normal", color=color)

    def arrow(x1, y1, x2, y2, color="#444", text=None, fontsize=7.5,
              text_offset=(0, 0.6), style="->"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.2))
        if text:
            ax.text((x1+x2)/2 + text_offset[0], (y1+y2)/2 + text_offset[1],
                    text, ha="center", va="center", fontsize=fontsize,
                    color=color, style="italic")

    # Stage 헤더
    ax.text(15, 49, "Stage 1: DL Fine-tuning",  fontsize=10, fontweight="bold",
            color="#1a3d6e", ha="center")
    ax.text(50, 49, "Stage 2: ML Stacking (with Transformer OOF)",
            fontsize=10, fontweight="bold", color="#1a3d6e", ha="center")
    ax.text(85, 49, "Stage 3: Ensemble",       fontsize=10, fontweight="bold",
            color="#1a3d6e", ha="center")

    # Stage 1: 5-fold ProteinBERT
    box(2, 38, 26, 7,
        "ProteinBERT (pretrained,\n92400 epoch / 23.5M sample)",
        "#1a3d6e", fc="#e8eef6", fontsize=8.5, bold=True)
    box(2, 28, 26, 7,
        "5-fold Fine-tuning\n(seed=42, lr=2e-3, seqlen=512)",
        "#1a3d6e", fc="white")
    box(2, 18, 26, 7,
        "fold_1/.h5 ... fold_5/.h5\n(checkpoints 5개)",
        "#1a3d6e", fc="white", fontsize=8)

    arrow(15, 38, 15, 35, color="#1a3d6e")
    arrow(15, 28, 15, 25, color="#1a3d6e")

    # OOF 산출 + Test 평균
    box(2, 5, 12.5, 9,
        "OOF preds\n(train 전체)\n→ train 메타",
        "#3d7a3d", fc="#e8f6e8", fontsize=7.8, bold=True)
    box(15.5, 5, 12.5, 9,
        "5-fold avg preds\n(test)\n→ test 메타",
        "#3d7a3d", fc="#e8f6e8", fontsize=7.8, bold=True)
    arrow(8, 18, 8, 14.5, color="#3d7a3d", text="val 부분만 추론", fontsize=6.5)
    arrow(22, 18, 22, 14.5, color="#3d7a3d", text="test 5번 추론 평균", fontsize=6.5)

    # Stage 2: ML Stacking
    box(33, 38, 34, 7,
        "ML 입력 (422-dim per sample):\n"
        "AAC(20) + DPC(400) + Length(1) + DL meta(1)",
        "#a64545", fc="#fbe9e9", fontsize=8, bold=True)
    arrow(28, 9.5, 33, 41.5, color="#3d7a3d", style="->",
          text="Train OOF meta\nTest avg meta", fontsize=7,
          text_offset=(2, -2))

    box(33, 24, 9.5, 8, "LightGBM\n(Optuna 100)", "#0072B2", fc="white", fontsize=8)
    box(45.3, 24, 9.5, 8, "XGBoost\n(Optuna 100)", "#E69F00", fc="white", fontsize=8)
    box(57.5, 24, 9.5, 8, "CatBoost\n(Optuna 100)", "#009E73", fc="white", fontsize=8)
    arrow(50, 38, 38, 32, color="#444", style="-")
    arrow(50, 38, 50, 32, color="#444", style="-")
    arrow(50, 38, 62, 32, color="#444", style="-")

    box(33, 12, 34, 8,
        "Best params 확정 → 최종 학습\n(lgbm.txt / xgb.json / catboost.cbm)",
        "#a64545", fc="white", fontsize=8)
    arrow(50, 24, 50, 20, color="#a64545")

    # Stage 3: Ensemble
    box(72, 30, 26, 8,
        "Soft-Voting Ensemble:\np = mean(p_lgbm, p_xgb, p_cat)",
        "#D55E00", fc="#fde8d8", fontsize=8.5, bold=True)
    arrow(67, 16, 72, 32, color="#a64545", style="->",
          text="3 trained\nmodels", fontsize=7, text_offset=(2, -1))

    box(72, 18, 26, 8,
        "Threshold = 0.5\nŷ = 1 if p ≥ 0.5 else 0",
        "#D55E00", fc="white", fontsize=8.5)
    arrow(85, 30, 85, 26, color="#D55E00")

    box(72, 6, 26, 8,
        "Final prediction:\nKla-site / Non-Kla-site",
        "#1a3d6e", fc="#e8eef6", fontsize=8.5, bold=True)
    arrow(85, 18, 85, 14, color="#D55E00")

    # 단계 구분 점선
    for x in [30, 70]:
        ax.plot([x, x], [3, 50], linestyle=":", color="#bbb", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PDF 본문
# ============================================================
BASE = "/home/work/LNP_TEST/git_tools/PBertKla_v2"
DIAGRAM_PNG = f"{BASE}/results/_arch_diagram_tmp.png"
draw_architecture(DIAGRAM_PNG)

story = []

# === Page 1 ===
story.append(p("PBertKla + ML 스태킹 모델 아키텍처", H_TITLE))
story.append(p("3-Stage Sequential Pipeline · DL Fine-tuning → ML Stacking → Soft-Voting Ensemble",
               H_SUB))

story.append(Image(DIAGRAM_PNG, width=17*cm, height=8.8*cm))
story.append(p("Figure 1. 전체 파이프라인. Stage 1의 transformer 5-fold OOF/평균이 "
               "Stage 2 ML 모델들의 메타 피처로 흐른다.", CAPTION))

story.append(h1("1. 개요"))
story.append(p(
    "본 모델은 단백질 서열로부터 라이신 락틸레이션(Kla) 부위를 예측하는 "
    "<b>3단계 sequential stacking</b> 파이프라인이다. Stage 1에서 ProteinBERT 백본을 "
    "5-fold 교차검증으로 파인튜닝해 강한 representation을 얻고, Stage 2에서 그 출력을 "
    "메타 피처로 받아 트리 기반 ML 모델 3종을 학습한다. Stage 3에서는 ML 3종의 출력을 "
    "soft-voting으로 결합해 최종 예측을 낸다."
))

# === Page 2 ===
story.append(PageBreak())
story.append(h1("2. Stage 1 — ProteinBERT 5-fold 파인튜닝"))
story.append(p(
    "단백질 서열을 입력으로 받아 Kla 가능성 확률(0~1)을 출력하는 DL 모델 단계. "
    "전체 train CSV를 StratifiedKFold(n_splits=5, shuffle=True, seed=42)로 분할하고 "
    "각 fold마다 ProteinBERT를 새로 파인튜닝한다."
))

story.append(h2("2.1 Backbone 모델"))
story.extend(bullets([
    "<b>ProteinBERT pretrained checkpoint</b>: epoch 92,400 / sample 23.5M",
    "Local + Global attention 결합 구조 (Conv-1D + Multi-head attention 6 blocks)",
    "Sigmoid 분류 head (2-class binary: Kla / Non-Kla)",
    "Dropout 0.5 (분류 head)",
]))

story.append(h2("2.2 학습 절차 (각 fold)"))
stage1_table = [
    ["Stage", "설명", "핵심 하이퍼파라미터"],
    ["A. Frozen warmup", "Pretrained transformer freeze, head만 학습",
     "lr=1e-2, frozen=True"],
    ["B. Full fine-tune", "전체 unfreeze, val_loss 기반 early stopping",
     "lr=2e-3, batch=32, seqlen=512, max_epochs=100, patience=15"],
    ["C. Final long-seq", "seqlen 1024로 1 epoch 추가 학습",
     "lr=1e-5, n_final_epochs=1"],
]
story.append(make_table(stage1_table, col_widths=[3.0*cm, 6.0*cm, 7.5*cm],
                        font_size=8.5))
story.append(p(
    "콜백: ModelCheckpoint(monitor=val_loss, save_best_only), "
    "ReduceLROnPlateau(factor=0.25, min_lr=1e-5), "
    "EarlyStopping(restore_best_weights=True). "
    "각 fold는 같은 공통 test CSV로 평가되어 fold_k/y_pred.npy로 저장된다.", NOTE))

story.append(h2("2.3 Stage 1 산출물"))
story.append(p(
    "fold_k 디렉토리(k=1..5) 각각에:"
))
story.extend(bullets([
    "<b>best_fine_tuning_model.h5</b> — 가중치 (Stage 2의 OOF 생성에 재활용)",
    "<b>y_pred.npy</b> — 공통 test set에 대한 예측 (Stage 2의 test 메타 피처 재료)",
    "<b>y_val_pred.npy / val_idx.npy</b> — 자기 fold의 validation 예측 (OOF 조립용)",
    "Train 전체에 대한 OOF 벡터 → 부모 디렉토리의 <b>oof_pred.npy</b>로 조립",
]))

# === Page 3 ===
story.append(PageBreak())
story.append(h1("3. Stage 2 — ML Stacking (Transformer OOF 메타)"))
story.append(p(
    "Stage 1에서 만든 transformer 출력을 메타 피처로 받아, 트리 기반 ML 모델 3종을 "
    "학습한다. 본 단계의 핵심은 <b>train과 test 시점의 메타 피처가 모두 transformer "
    "분포를 갖도록</b> 통일한 점이다 (정석 Stacked Generalization)."
))

story.append(h2("3.1 ML 입력 피처 구성 (총 422차원)"))
feat_table = [
    ["피처 그룹", "차원", "출처", "비고"],
    ["AAC (Amino Acid Composition)", "20",  "수기 통계", "20개 표준 아미노산 빈도 비율"],
    ["DPC (Dipeptide Composition)",  "400", "수기 통계", "20×20 dipeptide 빈도 비율"],
    ["Sequence length",              "1",   "수기 통계", "정수 길이"],
    ["DL Meta Feature",              "1",   "PBertKla 출력",
     "<b>Train: 5-fold OOF / Test: 5-fold avg</b>"],
    ["합계",                          "422", "—", "—"],
]
story.append(make_table(feat_table, col_widths=[5.0*cm, 1.5*cm, 3.5*cm, 6.5*cm],
                        font_size=8.5))

story.append(h2("3.2 메타 피처의 의미 — Train/Test 분포 일치"))
story.extend(bullets([
    "<b>Train 메타 (OOF)</b>: 각 train 샘플의 메타 피처를, 그 샘플을 학습에 사용하지 않은 "
    "fold에서 얻음. 5-fold이므로 모든 train 샘플이 정확히 한 번씩 \"본인을 학습에 안 본 모델\"의 "
    "예측을 받음. 데이터 누수 없음.",
    "<b>Test 메타 (5-fold avg)</b>: 5개 fold가 모두 학습에 사용하지 않은 test 샘플에 대해 "
    "각자 예측 → 5개 예측의 평균. 분산 감소 효과.",
    "<b>분포 일관성</b>: 두 단계 모두 \"transformer 출력\" 분포 → ML이 학습한 결합 규칙이 "
    "test에서 그대로 일반화됨.",
]))

story.append(h2("3.3 ML 모델 3종 + Optuna 튜닝"))
ml_table = [
    ["모델", "탐색 하이퍼파라미터", "튜닝 설정"],
    ["LightGBM",
     "n_estimators, learning_rate, max_depth, num_leaves, subsample, "
     "colsample_bytree, reg_alpha/lambda, min_child_samples",
     "100 trials × 5-fold CV"],
    ["XGBoost",
     "n_estimators, learning_rate, max_depth, subsample, colsample_bytree, "
     "gamma, reg_alpha/lambda, min_child_weight",
     "100 trials × 5-fold CV"],
    ["CatBoost",
     "iterations, learning_rate, depth, l2_leaf_reg, "
     "bagging_temperature, border_count",
     "100 trials × 5-fold CV"],
]
story.append(make_table(ml_table, col_widths=[2.5*cm, 9.0*cm, 4.5*cm],
                        font_size=8))
story.append(p(
    "Optuna TPE 샘플러 (seed=42)로 AUC-ROC를 maximize. 각 trial은 422차원 입력에 대해 "
    "5-fold StratifiedKFold CV의 평균 AUC를 평가. 최적 params로 train 전체 다시 학습 후 "
    "lgbm_model.txt / xgb_model.json / catboost_model.cbm 저장.", NOTE))

# === Page 4 ===
story.append(PageBreak())
story.append(h1("4. Stage 3 — Soft-Voting Ensemble"))

story.append(p(
    "Stage 2에서 학습한 3종 ML 모델의 예측 확률을 단순 평균하여 최종 확률을 계산하고, "
    "임계값 0.5를 적용해 binary 분류한다."
))
story.append(h2("4.1 수식"))
story.append(p(
    "<b>p̄ = (p<sub>lgbm</sub> + p<sub>xgb</sub> + p<sub>cat</sub>) / 3</b><br/>"
    "<b>ŷ = 1 (Kla-site) if p̄ ≥ 0.5, else 0 (Non-Kla-site)</b>", NOTE))

story.append(h2("4.2 설계 결정 — 왜 단순 평균인가"))
story.extend(bullets([
    "<b>학습 가중치 없음</b>: validation set이 작거나 유사할 때 weighted voting의 "
    "가중치가 overfit될 위험. 단순 평균은 편향 없이 robust.",
    "<b>DL은 ensemble에 직접 포함하지 않음</b>: DL 출력은 Stage 2에서 이미 메타 피처로 "
    "흡수되었으므로 별도 포함 시 double-counting. ML 3종이 각자 다른 방식으로 "
    "(DL 메타 + sequence) 정보를 결합한 결과를 평균.",
    "<b>3 모델의 다양성 가정</b>: LightGBM/XGBoost는 비슷한 boosting 구조이지만 "
    "split 알고리즘 차이로 약간의 다양성 제공, CatBoost는 ordered boosting과 "
    "categorical handling으로 더 큰 다양성 기여.",
]))

story.append(h1("5. 추론 시점의 데이터 흐름"))
infer_table = [
    ["단계", "입력", "처리", "출력"],
    ["1. DL 추론",
     "Test 서열 + 5-fold weight 5개",
     "각 fold weight로 test 추론 → 5개 예측 평균",
     "test 메타 피처 (1차원)"],
    ["2. Sequence feature 추출",
     "Test 서열",
     "AAC(20) + DPC(400) + length(1)",
     "421차원"],
    ["3. Concat",
     "421 + 1",
     "단순 hstack",
     "422차원 입력 X"],
    ["4. ML 추론",
     "X 422차원",
     "lgbm.txt, xgb.json, catboost.cbm 로드 후 predict_proba",
     "p_lgbm, p_xgb, p_cat (각 N차원)"],
    ["5. Soft voting",
     "3개 확률 벡터",
     "산술 평균",
     "p̄ (N차원)"],
    ["6. Threshold",
     "p̄",
     "≥ 0.5 → 1, else 0",
     "최종 binary 예측"],
]
story.append(make_table(infer_table,
                        col_widths=[2.4*cm, 4.0*cm, 6.5*cm, 3.6*cm],
                        font_size=8))

# === Page 5 ===
story.append(PageBreak())
story.append(h1("6. 핵심 설계 결정 요약"))
design_table = [
    ["결정 사항", "채택", "근거"],
    ["DL backbone", "ProteinBERT (epoch 92400)",
     "단백질 도메인 전문 pretrained, contextual embedding이 단순 통계보다 강력"],
    ["DL CV 폴드 수", "5-fold StratifiedKFold",
     "데이터 크기에 적정. 너무 작으면 분산↑, 너무 크면 학습 오래 걸림"],
    ["Train 메타 출처", "Transformer 5-fold OOF",
     "데이터 누수 없으면서 test와 같은 transformer 분포 보장"],
    ["Test 메타 출처", "Transformer 5-fold 평균",
     "단일 fold cherry-picking 회피, 분산 감소"],
    ["ML 모델 선택", "LightGBM + XGBoost + CatBoost",
     "Tree-based 다양성, Optuna 튜닝 효율성, 학습 속도"],
    ["Sequence feature", "AAC + DPC + length",
     "단순하고 빠름. 향후 ESM/conservation 등으로 교체 가능"],
    ["Optuna trials", "100 × 5-fold CV",
     "수렴에 충분. 50으로 줄여도 비슷한 성능"],
    ["Ensemble 방식", "단순 평균 (soft voting)",
     "weighted voting의 가중치 overfit 위험 회피"],
    ["분류 임계값", "0.5",
     "기본값. 검증셋에서 F1 최대화하는 threshold로 fine-tune 가능"],
]
story.append(make_table(design_table, col_widths=[3.5*cm, 4.5*cm, 8.5*cm],
                        font_size=8))

story.append(h1("7. 모델 파일 위치 (참조)"))
location_table = [
    ["산출물", "경로"],
    ["DL 가중치 (fold별)", "results/data{N}/fold_k/best_fine_tuning_model.h5"],
    ["DL OOF 벡터", "results/data{N}/oof_pred.npy"],
    ["DL fold별 test 예측", "results/data{N}/fold_k/y_pred.npy"],
    ["ML 모델", "results/ML_output/data{N}_oof/{lgbm_model.txt, xgb_model.json, catboost_model.cbm}"],
    ["ML 최적 hyperparameters", "results/ML_output/data{N}_oof/best_hyperparams.json"],
    ["내부 test 결과", "results/ML_output/data{N}_oof/results_summary.json"],
]
story.append(make_table(location_table, col_widths=[5.0*cm, 11.5*cm], font_size=8))

# 빌드
out_path = f"{BASE}/PBertKla_model_architecture.pdf"
doc = SimpleDocTemplate(
    out_path, pagesize=A4,
    leftMargin=1.6*cm, rightMargin=1.6*cm,
    topMargin=1.6*cm, bottomMargin=1.6*cm,
    title="PBertKla Model Architecture",
    author="lnpsolution@lnpsolution.com",
)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Nanum", 8)
    canvas.setFillColor(colors.HexColor("#888"))
    canvas.drawString(1.6*cm, 1.0*cm, "PBertKla 모델 아키텍처")
    canvas.drawRightString(A4[0]-1.6*cm, 1.0*cm, f"- {doc.page} -")
    canvas.restoreState()


doc.build(story, onFirstPage=footer, onLaterPages=footer)

# 임시 다이어그램 정리
if os.path.exists(DIAGRAM_PNG):
    os.remove(DIAGRAM_PNG)

print(f"✅ PDF 생성 완료: {out_path}")
print(f"   파일 크기: {os.path.getsize(out_path)/1024:.1f} KB")
