"""
Build a comprehensive professional PDF report for the Customer LTV & Churn
project, including FULL interpretation AND every line of code (SQL + Python).
Run:  python make_full_report.py
Output: report/Customer_LTV_Prediction_Full_Report.pdf
"""
import json
from pathlib import Path
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, PageBreak, Preformatted, ListFlowable, ListItem)

FIG = Path("outputs"); OUT = Path("report"); OUT.mkdir(exist_ok=True)
m = json.load(open(FIG / "metrics.json"))
AUTHOR = "Kingsley Amegah"
INK = colors.HexColor("#1a2330"); ACCENT = colors.HexColor("#1f78b4"); MUTED = colors.HexColor("#5b6573")

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10.5, leading=15.5,
                      alignment=TA_JUSTIFY, spaceAfter=8, textColor=colors.HexColor("#23292f"))
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, leading=19, textColor=INK,
                    spaceBefore=14, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, leading=16, textColor=ACCENT,
                    spaceBefore=10, spaceAfter=4)
cap = ParagraphStyle("cap", parent=ss["BodyText"], fontSize=8.5, leading=11, textColor=MUTED,
                     alignment=TA_CENTER, spaceBefore=3, spaceAfter=12)
code = ParagraphStyle("code", parent=ss["BodyText"], fontName="Courier", fontSize=7, leading=9,
                      textColor=colors.HexColor("#1a2330"), backColor=colors.HexColor("#f4f5f7"),
                      borderColor=colors.HexColor("#e0e3e8"), borderWidth=0.5, borderPadding=5,
                      spaceBefore=4, spaceAfter=12)

def P(t): return Paragraph(t, body)
def H1(t): return Paragraph(t, h1)
def H2(t): return Paragraph(t, h2)
def bullets(items):
    return ListFlowable([ListItem(Paragraph(t, body), leftIndent=6, value="•") for t in items],
                        bulletType="bullet", leftIndent=14, spaceAfter=8)
def figure(path, caption, max_w=6.6*inch, max_h=4.3*inch):
    iw, ih = PILImage.open(path).size; ar = iw/ih
    w = max_w; h = w/ar
    if h > max_h: h = max_h; w = h*ar
    img = Image(str(path), width=w, height=h); img.hAlign = "CENTER"
    return [img, Paragraph(caption, cap)]
def code_block(path, title):
    txt = Path(path).read_text()
    return [H2(title), Preformatted(txt, code)]

story = []

# ---- COVER ----
story += [Spacer(1, 1.4*inch),
    Paragraph("TECHNICAL PROJECT REPORT", ParagraphStyle("kc", parent=body, alignment=TA_CENTER,
              textColor=ACCENT, fontSize=11)), Spacer(1, 0.15*inch),
    Paragraph("Predicting Customer Lifetime Value and Churn for Marketing",
              ParagraphStyle("t", parent=ss["Title"], fontSize=24, leading=30, textColor=INK)),
    Spacer(1, 0.1*inch),
    Paragraph("A SQL and Python case study on 805,549 e-commerce transactions",
              ParagraphStyle("s", parent=ss["Title"], fontSize=13, leading=18, textColor=MUTED,
                             fontName="Helvetica")),
    Spacer(1, 1.5*inch),
    Paragraph(f"<b>{AUTHOR}</b>", ParagraphStyle("a", parent=ss["Title"], fontSize=15, textColor=INK)),
    Paragraph("Data Scientist", ParagraphStyle("r", parent=ss["Title"], fontSize=11,
              textColor=MUTED, fontName="Helvetica")), Spacer(1, 0.3*inch),
    Paragraph("SQL (SQLite) &nbsp;|&nbsp; Python &middot; scikit-learn &nbsp;|&nbsp; "
              "github.com/Kingsley-amg/customer-ltv-prediction",
              ParagraphStyle("f", parent=ss["Title"], fontSize=9.5, textColor=MUTED,
                             fontName="Helvetica")), PageBreak()]

# ---- EXEC SUMMARY ----
story.append(H1("Executive summary"))
story.append(P(
    "This project predicts which e-commerce customers are most valuable to a business in the "
    "coming quarter, so that marketing can target its budget for maximum return. It uses the "
    "public Online Retail II dataset, comprising <b>805,549 cleaned transactions</b> from a UK "
    f"online retailer (2009-2011) across <b>{m['n_customers']:,} customers</b>. Features are "
    "engineered in <b>SQL</b> and models are built in <b>Python</b>."))
story.append(bullets([
    f"A repeat-purchase classifier predicts whether a customer will buy again next quarter with "
    f"<b>ROC-AUC {m['churn_roc_auc']:.2f}</b>.",
    f"Targeting the <b>top 20% of customers by predicted value</b> captures "
    f"<b>{m['model_capture_top20']*100:.0f}% of next-quarter revenue</b> - a "
    f"<b>{m['lift_top20_vs_random']:.1f}x lift</b> over untargeted outreach.",
    f"The highest predicted-value decile spends about <b>{m['top_decile_vs_bottom_decile_x']:.0f}x</b> "
    "more than the lowest, confirming the score concentrates value.",
    f"The model surfaces <b>{m['n_at_risk_test']} high-value customers</b> worth "
    f"<b>GBP {m['revenue_at_risk_test_gbp']:,.0f}</b> of historic revenue (in a 25% hold-out; "
    f"roughly GBP {m['revenue_at_risk_test_gbp']*4/1e6:.1f}M across the full base) who are "
    "predicted to lapse - a ready-made retention list.",
    "Honest caveat: for pure revenue ranking, sorting by past spend is a strong baseline the "
    "model only matches; the model's distinct value is the per-customer churn probability."]))

# ---- 1. BACKGROUND ----
story.append(H1("1. Background and objective"))
story.append(P(
    "Marketing budgets are finite, so the practical question is not 'who are our customers?' but "
    "'where should we spend to protect the most future revenue?'. Answering it requires "
    "predicting <b>future</b> customer behaviour and value, not just summarising the past."))
story.append(P(
    "<b>Objective:</b> from each customer's purchasing history up to a cutoff date, predict "
    "(a) whether they will purchase again in the next quarter and (b) how much they will spend, "
    "then convert those predictions into concrete marketing actions with a measurable revenue impact."))

# ---- 2. DATA ----
story.append(H1("2. Data and feature engineering"))
story.append(P(
    "The raw data is 1,067,371 transaction lines; after removing cancellations, returns, missing "
    "customer IDs and non-positive prices, <b>805,549 clean purchase lines</b> remain. These were "
    "loaded into a <b>SQLite database</b>, and SQL was used to build a customer-level feature "
    "table. The design follows a calibration / prediction split: features are computed from all "
    "activity <b>before</b> a cutoff date (2011-09-09), and the target is each customer's revenue "
    "in the <b>91 days after</b> it."))
story.append(bullets([
    "<b>Recency</b> - days since the customer's last purchase before the cutoff.",
    "<b>Frequency</b> - number of distinct orders; <b>tenure</b> - days since first purchase.",
    "<b>Monetary</b> - total and average order value; <b>recent_revenue_90d</b> - spend in the "
    "last 90 days of the calibration window (a momentum signal).",
    "<b>Breadth</b> - distinct products bought, total items, and number of active months.",
    f"<b>Target</b> - next-quarter revenue (and its binary form, 'active next quarter', true for "
    f"{m['active_next_quarter_pct']*100:.0f}% of customers)."]))
story.append(P("The exact SQL is listed in Appendix A."))

# ---- 3. METHODOLOGY ----
story.append(H1("3. Methodology"))
story.append(P(
    "The customer table was split 75/25 into training and test sets (stratified on the active "
    "flag). Two complementary models were trained on the same features:"))
story.append(bullets([
    "<b>Behaviour model</b> - a gradient-boosted classifier predicting the probability that a "
    "customer purchases next quarter.",
    "<b>Value model</b> - a gradient-boosted regressor predicting next-quarter spend, trained on "
    "the log of revenue because the target is heavily skewed and zero-inflated (Figure 1).",
    "The two are combined into an <b>expected-value score</b> (probability of buying multiplied by "
    "predicted spend), which is used to rank customers for targeting."]))
story += figure(FIG / "01_target_distribution.png",
                "Figure 1. Next-quarter revenue is skewed and zero-inflated, motivating a log-scale "
                "value model and a separate behaviour (buy / no-buy) model.", max_h=3.0*inch)

# ---- 4. RESULTS ----
story.append(H1("4. Results and interpretation"))

story.append(H2("4.1 Predicting behaviour (will the customer return?)"))
story.append(P(
    f"The classifier reaches <b>ROC-AUC {m['churn_roc_auc']:.2f}</b> (PR-AUC {m['churn_pr_auc']:.2f}) "
    "on the held-out customers, meaning it reliably separates customers who will return from those "
    "who will lapse. This is the engine behind the retention use case in Section 5."))
story += figure(FIG / "04_churn_roc.png",
                "Figure 2. ROC curve for the repeat-purchase model.", max_h=3.2*inch)
story.append(P(
    "Permutation importance shows the prediction is driven mainly by <b>recency</b> and "
    "<b>recent 90-day spend</b>, followed by frequency and the number of active months - i.e. "
    "how recently and how consistently a customer has engaged, which matches retail intuition."))
story += figure(FIG / "05_feature_importance.png",
                "Figure 3. What predicts a repeat purchase (permutation importance).", max_h=3.2*inch)

story.append(H2("4.2 Predicting value and the marketing impact"))
story.append(P(
    f"Ranking customers by the expected-value score and contacting only the top 20% would reach "
    f"<b>{m['model_capture_top20']*100:.0f}% of all next-quarter revenue</b>, versus 20% from "
    f"random outreach - a <b>{m['lift_top20_vs_random']:.1f}x lift</b>. The gains curve (Figure 4) "
    "sits far above the random diagonal. It also sits essentially on top of the 'rank by past "
    f"spend' baseline ({m['pastspend_capture_top20']*100:.0f}%), an honest and important finding: "
    "for revenue ranking alone, past spend is already an excellent and much simpler signal."))
story += figure(FIG / "02_gains_chart.png",
                "Figure 4. Gains chart. Targeting by predicted value (red) captures the majority of "
                "revenue from a small fraction of customers.", max_h=3.6*inch)
story.append(P(
    f"Grouping the test customers into deciles of predicted value confirms the ranking is "
    f"monotonic: the top decile's mean actual next-quarter revenue is about "
    f"<b>{m['top_decile_vs_bottom_decile_x']:.0f} times</b> the bottom decile's (Figure 5)."))
story += figure(FIG / "03_decile_lift.png",
                "Figure 5. Mean actual next-quarter revenue rises sharply across predicted-value "
                "deciles.", max_h=2.8*inch)

# ---- 5. BUSINESS IMPACT ----
story.append(H1("5. Business impact and recommended actions"))
story.append(bullets([
    f"<b>Concentrate spend.</b> Roughly two-thirds of next-quarter revenue comes from the top 20% "
    "of customers the model identifies - marketing can focus acquisition-style budgets there "
    "instead of spreading them evenly.",
    f"<b>Run a targeted retention campaign.</b> The model flags "
    f"<b>{m['n_at_risk_test']} historically high-value customers</b> (spend above the "
    f"GBP {m['high_value_threshold_gbp']:,.0f} 75th-percentile) who are predicted to lapse, "
    f"together worth <b>GBP {m['revenue_at_risk_test_gbp']:,.0f}</b> in the 25% hold-out alone "
    f"(about GBP {m['revenue_at_risk_test_gbp']*4/1e6:.1f}M across the full base). This is a "
    "named, prioritised list that a static past-spend report cannot produce.",
    "<b>Choose the simpler tool where it wins.</b> Because past spend ranks revenue almost as well, "
    "the model earns its place specifically through the churn probability; for pure value ranking a "
    "transparent RFM rule is a defensible, cheaper option."]))

# ---- 6. LIMITATIONS ----
story.append(H1("6. Limitations"))
story.append(bullets([
    "A single retailer over two years; patterns may not transfer to other businesses or periods.",
    "The 'at-risk revenue' figure uses historic spend as a proxy for value at stake, not a formal "
    "expected-loss calculation.",
    "The model captures association, not causation - contacting a customer is not guaranteed to "
    "change their behaviour; an A/B test would be the proper next step.",
    "Predicted spend amounts are approximate (the target is noisy and zero-inflated); the ranking "
    "is more reliable than the absolute predictions."]))

# ---- 7. CONCLUSION ----
story.append(H1("7. Conclusion"))
story.append(P(
    "Using SQL for feature engineering and Python for modelling, this project turns two years of "
    "raw transactions into forward-looking, actionable customer intelligence: who will return "
    f"(ROC-AUC {m['churn_roc_auc']:.2f}), who is most valuable (top 20% capture "
    f"{m['model_capture_top20']*100:.0f}% of next-quarter revenue), and which valuable customers "
    "are slipping away. The analysis is transparent about where a simple heuristic already suffices "
    "and where the model genuinely adds value. A natural next step is an A/B test of a "
    "model-guided retention campaign to measure causal uplift."))

# ---- APPENDICES: FULL CODE ----
story.append(PageBreak())
story.append(H1("Appendix A - SQL feature engineering (sql/build_features.sql)"))
story.append(Preformatted(Path("sql/build_features.sql").read_text(), code))
story.append(PageBreak())
story += code_block("01_build_features.py", "Appendix B - Data engineering script (01_build_features.py)")
story.append(PageBreak())
story += code_block("02_model_ltv.py", "Appendix C - Modelling script (02_model_ltv.py)")

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d6dbe0")); canvas.setLineWidth(0.5)
    canvas.line(0.8*inch, 0.7*inch, letter[0]-0.8*inch, 0.7*inch)
    canvas.setFont("Helvetica", 8); canvas.setFillColor(MUTED)
    canvas.drawString(0.8*inch, 0.55*inch, f"Customer LTV & Churn Prediction  |  {AUTHOR}")
    canvas.drawRightString(letter[0]-0.8*inch, 0.55*inch, f"Page {doc.page}")
    canvas.restoreState()

doc = SimpleDocTemplate(str(OUT / "Customer_LTV_Prediction_Full_Report.pdf"), pagesize=letter,
    leftMargin=0.8*inch, rightMargin=0.8*inch, topMargin=0.8*inch, bottomMargin=0.9*inch,
    title="Customer LTV & Churn Prediction - Technical Report", author=AUTHOR,
    subject="SQL + Python customer analytics project report", creator=AUTHOR)
doc.build(story, onLaterPages=footer)
print("Wrote", OUT / "Customer_LTV_Prediction_Full_Report.pdf")
