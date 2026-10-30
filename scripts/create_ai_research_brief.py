from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "presentation" / "ai_research_brief.pdf"


def p(text, style):
    return Paragraph(text, style)


def table(data, widths):
    result = Table(data, colWidths=widths, repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B5D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F1F5F9")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return result


def build():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER,
        fontSize=23, leading=28, textColor=colors.HexColor("#123B5D"),
    ))
    styles.add(ParagraphStyle(
        name="Subtitle", parent=styles["Normal"], alignment=TA_CENTER,
        fontSize=11, leading=15, textColor=colors.HexColor("#475569"),
    ))
    styles.add(ParagraphStyle(
        name="H", parent=styles["Heading2"], fontSize=15, leading=18,
        textColor=colors.HexColor("#123B5D"), spaceBefore=11, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="H3", parent=styles["Heading3"], fontSize=11.5, leading=14,
        textColor=colors.HexColor("#0F766E"), spaceBefore=7, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="Body", parent=styles["BodyText"], fontSize=9.4, leading=13,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="Small", parent=styles["BodyText"], fontSize=8, leading=10,
        textColor=colors.HexColor("#475569"),
    ))

    s = []
    s += [
        p("AI Research Handoff Brief", styles["TitleCenter"]),
        Spacer(1, 0.12 * inch),
        p("Paper ECG digitization, classwise calibration, and probability correction", styles["Subtitle"]),
        Spacer(1, 0.2 * inch),
        p("<b>Purpose:</b> This document is a self-contained, evidence-grounded summary for an AI or researcher joining the project. It explains what the repository does, what was executed, what improved, what is verified, and what must not be claimed.", styles["Body"]),
        p("<b>Repository:</b> Jaykvankar/paper_ecg_uncertainty | Working branch: <b>updatedversion</b>", styles["Body"]),
        p("<b>Primary conclusion:</b> Tracing a synthetic ECG paper page changes probability calibration in a diagnosis-specific way. The effect is much larger for the raw-millivolt model than for the standardized model, especially for conduction disturbance. Validation-fitted recalibration reduces the observed probability error to approximately zero.", styles["Body"]),
        p("Status legend: <b>Verified</b> means supported by saved results; <b>secondary</b> means not the primary estimand; <b>blocked</b> means not completed and must not be presented as a result.", styles["Small"]),
        PageBreak(),
    ]

    s += [
        p("1. Research problem", styles["H"]),
        p("A digital ECG can be printed on paper, scanned or photographed, and converted back into a digital waveform. The conversion introduces wrinkles, creases, missing ink, scale changes, and reconstruction artifacts. A classifier may still rank diagnoses reasonably well while its reported probabilities become unreliable.", styles["Body"]),
        p("The project asks: <b>Does tracing a synthetic PTB-XL paper page change classwise calibration, and does that change differ by diagnosis?</b>", styles["Body"]),
        p("This is not primarily a new-classifier project. The compact residual classifier is a controlled measuring instrument. The scientific target is the change in expected calibration error (ECE) between the original clean waveform and the traced waveform.", styles["Body"]),
        p("Calibration interpretation: if a model predicts 0.90 for many cases, approximately 90% of those cases should be positive. ECE summarizes the gap between predicted probabilities and observed frequencies. Positive ΔECE means tracing made calibration worse.", styles["Body"]),
        p("2. Scope and estimands", styles["H"]),
        table([
            ["Item", "Definition"],
            ["Primary outcome", "Classwise ΔECE = ECE(traced creased page) − ECE(clean waveform)."],
            ["Primary page", "Synthetic creased PTB-XL page rendered with ECG-Image-Kit."],
            ["Classes", "NORM, MI, STTC, CD, HYP."],
            ["Secondary uncertainty", "Change in mean binary entropy, in nats."],
            ["Companion discrimination", "Change in classwise AUROC."],
            ["Reconstruction quality", "PhysioNet Challenge 2024 lead-wise SNR, allowing alignment shifts."],
            ["Uncertainty", "Patient-cluster bootstrap, 1,000 resamples, seed 42, 2.5th/97.5th percentiles."],
        ], [1.65 * inch, 5.65 * inch]),
        PageBreak(),
    ]

    s += [
        p("3. Data and labels", styles["H"]),
        p("Dataset: PTB-XL 1.0.3, using 100 Hz recordings (12 leads, 1,000 samples, approximately 10 seconds). Diagnostic SCP statements are mapped to five multi-label superclasses: NORM (normal), MI (myocardial infarction), STTC (ST/T change), CD (conduction disturbance), and HYP (hypertrophy).", styles["Body"]),
        p("The data pipeline validates SHA-256 files, lead order, sampling rate, sample count, units, and finite values. Patient separation is checked across splits. Records without a diagnostic superclass are retained in the Stage 1 manifest but excluded from supervised Stage 2 training/evaluation.", styles["Body"]),
        table([
            ["Split", "Use", "Retained labeled records", "Unlabeled excluded"],
            ["Folds 1–8", "Training", "17,084", "334"],
            ["Fold 9", "Validation / checkpoint selection / correction fitting", "2,146", "37"],
            ["Fold 10", "One final paired test score", "2,158", "40"],
        ], [1.1 * inch, 3.7 * inch, 1.25 * inch, 1.25 * inch]),
        p("4. Model and controlled arms", styles["H"]),
        p("Both arms use the same compact one-dimensional residual network, binary cross-entropy with logits, AdamW, cosine learning-rate schedule, batch size 64, 30 epochs, seed 42, and fold-9 macro-AUROC checkpoint selection. No test-set tuning is allowed.", styles["Body"]),
        table([
            ["Arm", "Input representation", "Config/output"],
            ["Standardized", "Each lead is z-scored over time; absolute voltage scale is removed.", "configs/stage2.yaml → results/stage2"],
            ["Raw millivolt", "Absolute millivolt values are preserved.", "configs/stage2_raw.yaml → results/stage2_raw"],
        ], [1.2 * inch, 3.7 * inch, 2.4 * inch]),
        PageBreak(),
    ]

    s += [
        p("5. Complete experimental pipeline", styles["H"]),
        p("<b>Step 1 — Validate data:</b> Download/use PTB-XL, verify file hashes and waveform properties, map SCP statements, and create leakage-safe patient splits.", styles["Body"]),
        p("<b>Step 2 — Train clean baselines:</b> Train the standardized and raw-millivolt arms. Select checkpoints only on fold-9 macro-AUROC. Fold 10 remains untouched until final scoring.", styles["Body"]),
        p("<b>Step 3 — Render paper pages:</b> Render each test waveform using ECG-Image-Kit at the locked commit. The primary condition is one-column, 200 dpi, red grid, wrinkles, three vertical and three horizontal 45-degree creases, without header, lead names, calibration pulse, or handwriting.", styles["Body"]),
        p("<b>Step 4 — Digitize:</b> Use lead boxes and a dark-run column max-finder to recover the waveform. The plotted image pixels are not used as the recovered signal.", styles["Body"]),
        p("<b>Step 5 — Quality gate:</b> Trace a training record first. Every lead must correlate at least 0.80 with the clean lead before fold-10 paired scoring is permitted.", styles["Body"]),
        p("<b>Step 6 — Paired score:</b> Run each frozen model on the clean and traced versions of the same fold-10 records. Compute classwise ECE, AUROC, entropy, and SNR. Store patient-bootstrap intervals.", styles["Body"]),
        p("<b>Step 7 — Investigate mechanism:</b> Compare SNR by diagnosis, perform oracle scale matching, fit validation-only logit and trace-aware correction maps, and run a pre-specified width-64 sensitivity.", styles["Body"]),
        p("6. Key repository evidence", styles["H"]),
        table([
            ["Evidence", "Location"],
            ["Locked question and estimands", "PROTOCOL.md and configs/paired.yaml"],
            ["Pipeline and usage", "README.md"],
            ["Standardized paired result", "results/paired_standardized.json"],
            ["Raw-mV paired result", "results/paired_raw_mv.json"],
            ["Clean baselines", "results/stage2/REPORT.md and results/stage2_raw/REPORT.md"],
            ["Correction outputs", "results/corrector/raw_mv.json and logit_only.json"],
            ["SNR by diagnosis", "results/mechanism/snr_by_class.json"],
            ["Width-64 sensitivity", "results/sensitivity_wide/summary.json"],
            ["External scan gate", "results/real_scan/gate.json"],
        ], [2.4 * inch, 4.9 * inch]),
        PageBreak(),
    ]

    s += [
        p("7. Verified results", styles["H"]),
        p("Clean digital baselines:", styles["H3"]),
        table([
            ["Arm", "Clean macro-AUROC", "Clean ECE", "Clean Brier"],
            ["Standardized", "0.9076", "0.0398", "0.0913"],
            ["Raw millivolt", "0.9259", "0.0309", "0.0839"],
        ], [2.0 * inch, 1.7 * inch, 1.5 * inch, 1.5 * inch]),
        p("Primary creased-page ΔECE:", styles["H3"]),
        table([
            ["Class", "Standardized", "Raw millivolt", "Raw − standardized"],
            ["NORM", "+0.056", "+0.066", "+0.010"],
            ["MI", "+0.075", "+0.077", "+0.002"],
            ["STTC", "+0.005", "+0.041", "+0.037"],
            ["CD", "−0.019", "<b>+0.142</b>", "<b>+0.161</b>"],
            ["HYP", "+0.034", "+0.068", "+0.034"],
        ], [1.7 * inch, 1.8 * inch, 1.8 * inch, 2.0 * inch]),
        p("The CD raw-minus-standardized contrast has a patient-bootstrap interval of approximately 0.136–0.171, excluding zero. Mean reconstruction SNR is 9.06 dB in both primary arms.", styles["Body"]),
        p("8. What exactly improved?", styles["H"]),
        p("<b>Not clean classification accuracy:</b> the project did not claim a higher clean AUROC. In fact, traced-page AUROC generally decreased. The work improves understanding and reliability of the deployment pipeline, not the classifier's raw discrimination.", styles["Body"]),
        p("<b>Probability calibration was improved after correction:</b> the raw-millivolt CD shift was approximately +0.142 before correction. A validation-fitted logit-only map moved the CD shift to approximately −0.006; a trace-aware map moved it to approximately −0.005. These maps were fitted on fold 9 and applied once to fold 10.", styles["Body"]),
        p("<b>Scientific understanding improved:</b> we identified a large diagnosis-specific vulnerability, showed that it depends on voltage representation, tested whether reconstruction quality alone explained it, tested amplitude matching, and reproduced the direction/magnitude with a wider model.", styles["Body"]),
        PageBreak(),
    ]

    s += [
        p("9. Mechanism and robustness checks", styles["H"]),
        p("The CD effect is not simply explained by worse CD reconstruction. CD-labeled records had mean SNR 8.83 dB versus 9.13 dB without CD. Oracle matching of traced leads to the clean mean and standard deviation reduced the raw-mV CD shift but left approximately 0.125, so amplitude mismatch is only part of the mechanism.", styles["Body"]),
        p("A width-64 sensitivity model reproduced the main contrast: raw-minus-standardized CD ΔECE approximately 0.154, with an interval approximately 0.136–0.169. The pre-specified agreement rule was satisfied because the interval excluded zero and retained the same sign as the locked primary contrast.", styles["Body"]),
        p("Entropy also changed, especially for raw-mV CD (approximately +0.187 nats), showing that the digitization path changes the model's uncertainty behavior, not only its binary decision ranking.", styles["Body"]),
        p("10. Novelty and contribution", styles["H"]),
        p("The novelty is not a new network or a claim that digitization improves AUROC. It is the controlled paired study of <b>classwise probability calibration after ECG paper digitization</b>, with the preprocessing representation treated as an experimental factor. The work shows that raw millivolt inputs can be more calibration-sensitive than standardized inputs, even when clean performance is strong.", styles["Body"]),
        p("The project also provides a practical deployment lesson: a classifier intended to consume digitized ECGs should be calibrated on digitized ECGs, not only on clean digital waveforms. A simple validation-only recalibration map can substantially reduce the shift.", styles["Body"]),
        p("11. Claims that are safe versus unsafe", styles["H"]),
        table([
            ["Safe, supported claim", "Do not claim"],
            ["Synthetic creased pages caused a large CD calibration shift in the raw-mV arm.", "Clinical safety or deployment readiness."],
            ["The raw-vs-standardized CD contrast was reproduced by a wider model.", "Universal failure for every ECG model or digitizer."],
            ["Validation-fitted recalibration reduced the observed CD shift near zero.", "That recalibration is clinically validated."],
            ["The result is based on paired fold-10 PTB-XL evaluation.", "Generalization to phone photographs or thermal paper."],
            ["Mean page reconstruction SNR was about 9.06 dB.", "That SNR alone proves clinical waveform fidelity."],
        ], [3.65 * inch, 3.65 * inch]),
        PageBreak(),
    ]

    s += [
        p("12. Current limitations and unfinished work", styles["H"]),
        p("The primary result uses one seed, a compact architecture, and synthetic creases. The external color-scan experiment is blocked because its required variant identifier is unset; no fold-10 color-scan result should be reported. The claim does not cover phone photographs, Emory records, Akershus thermal paper, mold, or clinical thresholds.", styles["Body"]),
        p("The repository intentionally excludes large image/PDF/NPZ artifacts from the pushed branch while retaining result summaries, reports, source, configurations, and tests.", styles["Body"]),
        p("13. Recommended context for future AI work", styles["H"]),
        p("Before editing or extending the project, read PROTOCOL.md, README.md, CHANGELOG.md, and the locked JSON result files. Preserve the primary estimand, the creased-page condition, the fold split, the one-score-per-arm rule, and the distinction between primary and secondary outcomes.", styles["Body"]),
        p("Do not overwrite locked primary result files merely to make a rerun look better. Do not tune on fold 10. Do not convert a secondary corrector or sensitivity result into a new primary claim without an explicit protocol update.", styles["Body"]),
        p("Potential next steps are: set and document the external scan variant id, pass the training-fold color-scan gate, score the external scan once per frozen model, add more seeds, quantify calibration confidence intervals beyond the existing paired bootstrap, and validate correction on an independent dataset.", styles["Body"]),
        p("14. Final one-paragraph summary", styles["H"]),
        p("This repository studies whether converting digital ECGs to synthetic paper pages and tracing them back changes the reliability of diagnostic probabilities. Using PTB-XL and two identical models—one trained on standardized leads and one on raw millivolts—we found that creased-page tracing produced a particularly large conduction-disturbance calibration shift for the raw-mV model (+0.142), while the standardized model changed by −0.019. The raw-minus-standardized contrast was approximately +0.161 with a bootstrap interval excluding zero, was reproduced by a wider network, and was not explained solely by trace SNR or amplitude mismatch. A validation-only recalibration map reduced the CD shift near zero. The contribution is therefore a diagnosis-specific, reproducible calibration analysis of ECG paper digitization, with a practical correction strategy and clearly bounded synthetic-page scope.", styles["Body"]),
    ]

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, rightMargin=0.58 * inch, leftMargin=0.58 * inch,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        title="AI Research Handoff Brief",
        author="Paper ECG Uncertainty project",
    )
    doc.build(s)
    return OUT


if __name__ == "__main__":
    print(build())
