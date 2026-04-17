import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    Table, TableStyle
)
from reportlab.lib.enums import TA_LEFT


# ── Color palette (light theme for PDF readability) ────────────────────────────
ACCENT      = colors.HexColor("#b8860b")   # dark gold
TEXT_MAIN   = colors.HexColor("#1a1a1a")   # near black
TEXT_MUTED  = colors.HexColor("#555555")   # medium gray
BORDER      = colors.HexColor("#dddddd")   # light gray
ACCENT_LIGHT= colors.HexColor("#f5f0e8")   # warm cream background
RED         = colors.HexColor("#c0392b")
YELLOW      = colors.HexColor("#d68910")
GREEN       = colors.HexColor("#1e8449")


def get_risk_color(risk_label: str):
    if "High" in risk_label:
        return RED
    elif "Medium" in risk_label:
        return YELLOW
    return GREEN


def generate_compliance_pdf(
    question: str,
    answer: str,
    specialist: str = "",
    risk_label: str = "",
    citations: list = None,
) -> bytes:
    """
    Generate a formatted compliance report PDF.
    Returns PDF as bytes for Streamlit download_button.
    """
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
    )

    # ── Styles ─────────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "Title",
        fontName="Times-Bold",
        fontSize=22,
        textColor=TEXT_MAIN,
        spaceAfter=2,
        alignment=TA_LEFT,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        fontName="Helvetica",
        fontSize=8.5,
        textColor=TEXT_MUTED,
        spaceAfter=2,
        leading=13,
    )

    section_label_style = ParagraphStyle(
        "SectionLabel",
        fontName="Helvetica-Bold",
        fontSize=7,
        textColor=ACCENT,
        spaceBefore=18,
        spaceAfter=6,
        leading=10,
    )

    question_style = ParagraphStyle(
        "Question",
        fontName="Times-Roman",
        fontSize=14,
        textColor=TEXT_MAIN,
        spaceAfter=8,
        leading=22,
    )

    body_style = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=10,
        textColor=TEXT_MAIN,
        spaceAfter=6,
        leading=17,
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        fontName="Helvetica",
        fontSize=10,
        textColor=TEXT_MAIN,
        spaceAfter=4,
        leading=16,
        leftIndent=16,
        bulletIndent=0,
    )

    meta_key_style = ParagraphStyle(
        "MetaKey",
        fontName="Helvetica",
        fontSize=8.5,
        textColor=TEXT_MUTED,
    )

    meta_val_style = ParagraphStyle(
        "MetaVal",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        textColor=TEXT_MAIN,
    )

    muted_style = ParagraphStyle(
        "Muted",
        fontName="Helvetica",
        fontSize=8.5,
        textColor=TEXT_MUTED,
        spaceAfter=3,
        leading=14,
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        fontName="Helvetica-Oblique",
        fontSize=8,
        textColor=TEXT_MUTED,
        spaceBefore=10,
        leading=13,
    )

    story = []
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    # ── Header ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("AI Compliance Assistant", title_style))
    story.append(Paragraph("Compliance Report", subtitle_style))
    story.append(Paragraph(f"Generated: {timestamp}", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT, spaceAfter=14))

    # ── Metadata table ─────────────────────────────────────────────────────────
    meta_rows = []
    if specialist:
        clean_spec = specialist.strip()
        for emoji in ["⚖️","🏥","🇪🇺","🤖","💰","🔒"]:
            clean_spec = clean_spec.replace(emoji, "").strip()
        meta_rows.append([
            Paragraph("SPECIALIST", meta_key_style),
            Paragraph(clean_spec, meta_val_style)
        ])
    if risk_label:
        clean_risk = risk_label.replace("🔴","").replace("🟡","").replace("🟢","").strip()
        risk_color = get_risk_color(risk_label)
        risk_val_style = ParagraphStyle("RiskVal", fontName="Helvetica-Bold",
                                        fontSize=8.5, textColor=risk_color)
        meta_rows.append([
            Paragraph("COMPLIANCE RISK", meta_key_style),
            Paragraph(clean_risk, risk_val_style)
        ])

    if meta_rows:
        t = Table(meta_rows, colWidths=[1.5 * inch, 5.2 * inch])
        t.setStyle(TableStyle([
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LINEBELOW",     (0,0), (-1,-2), 0.5, BORDER),
            ("BACKGROUND",    (0,0), (-1,-1), colors.white),
        ]))
        story.append(t)
        story.append(Spacer(1, 14))

    # ── Question ───────────────────────────────────────────────────────────────
    story.append(Paragraph("QUESTION", section_label_style))
    story.append(Paragraph(question, question_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=10))

    # ── Answer ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("ANSWER", section_label_style))

    clean_answer = answer.strip()
    lines = clean_answer.split("\n")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 5))
            continue
        # Skip disclaimer lines
        if stripped.startswith("⚠️") or "informational purposes only" in stripped.lower():
            continue
        # Skip markdown horizontal rules
        if stripped.startswith("---"):
            continue
        # Bullets
        if stripped.startswith(("- ", "• ", "* ")):
            story.append(Paragraph(f"&bull;&nbsp; {stripped[2:]}", bullet_style))
        # Numbered items like "1. something"
        elif len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".):":
            story.append(Paragraph(stripped, bullet_style))
        # Bold headers like "**Title**:"
        elif stripped.startswith("**") and stripped.endswith("**"):
            clean = stripped.replace("**", "")
            story.append(Paragraph(f"<b>{clean}</b>", body_style))
        else:
            # Strip inline markdown bold
            clean = stripped.replace("**", "")
            story.append(Paragraph(clean, body_style))

    # ── Sources ────────────────────────────────────────────────────────────────
    if citations:
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=6))
        story.append(Paragraph("SOURCES", section_label_style))
        seen = set()
        for c in citations:
            source = c.get("source", "Unknown")
            # Skip full path duplicates
            if source.startswith("data/"):
                continue
            page_str = f"Page {int(c['page']) + 1}" if c.get("page") is not None else "Page N/A"
            key = f"{source}_{page_str}"
            if key not in seen:
                seen.add(key)
                story.append(Paragraph(f"{source} — {page_str}", muted_style))

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 18))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=8))
    story.append(Paragraph(
        "This report is for informational purposes only and does not constitute legal advice. "
        "Please consult a qualified legal or compliance professional for guidance specific to your situation.",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()



def get_risk_color(risk_label: str):
    if "High" in risk_label:
        return RED
    elif "Medium" in risk_label:
        return YELLOW
    return GREEN


def generate_compliance_pdf(
    question: str,
    answer: str,
    specialist: str = "",
    risk_label: str = "",
    citations: list = None,
) -> bytes:
    """
    Generate a formatted compliance report PDF.
    Returns PDF as bytes for Streamlit download_button.
    """
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    # ── Styles ─────────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        fontName="Times-Roman",
        fontSize=22,
        textColor=TEXT_MAIN,
        spaceAfter=4,
        alignment=TA_LEFT,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        fontName="Helvetica",
        fontSize=8,
        textColor=TEXT_MUTED,
        spaceAfter=2,
        alignment=TA_LEFT,
        leading=12,
    )

    section_label_style = ParagraphStyle(
        "SectionLabel",
        fontName="Helvetica",
        fontSize=7,
        textColor=ACCENT,
        spaceBefore=16,
        spaceAfter=6,
        leading=10,
    )

    question_style = ParagraphStyle(
        "Question",
        fontName="Times-Roman",
        fontSize=13,
        textColor=TEXT_MAIN,
        spaceAfter=8,
        leading=20,
    )

    body_style = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=9.5,
        textColor=TEXT_MAIN,
        spaceAfter=6,
        leading=16,
    )

    muted_style = ParagraphStyle(
        "Muted",
        fontName="Helvetica",
        fontSize=8,
        textColor=TEXT_MUTED,
        spaceAfter=4,
        leading=13,
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        textColor=TEXT_MUTED,
        spaceBefore=12,
        leading=12,
        alignment=TA_LEFT,
    )

    # ── Build story ────────────────────────────────────────────────────────────
    story = []
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    # Header
    story.append(Paragraph("AI Compliance Assistant", title_style))
    story.append(Paragraph("Compliance Report", subtitle_style))
    story.append(Paragraph(f"Generated: {timestamp}", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=16))

    # Metadata row — specialist + risk
    meta_data = []
    if specialist:
        meta_data.append(["Specialist", specialist.replace("⚖️", "").replace("🏥", "").replace("🇪🇺", "").replace("🤖", "").replace("💰", "").replace("🔒", "").strip()])
    if risk_label:
        clean_risk = risk_label.replace("🔴", "").replace("🟡", "").replace("🟢", "").strip()
        meta_data.append(["Compliance Risk", clean_risk])

    if meta_data:
        table = Table(meta_data, colWidths=[1.4 * inch, 5 * inch])
        risk_color = get_risk_color(risk_label) if risk_label else TEXT_MUTED
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (0, -1), TEXT_MUTED),
            ("TEXTCOLOR", (1, 0), (1, -1), TEXT_MAIN),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, BORDER),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))

    # Question
    story.append(Paragraph("QUESTION", section_label_style))
    story.append(Paragraph(question, question_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=12))

    # Answer
    story.append(Paragraph("ANSWER", section_label_style))

    # Split answer into paragraphs and render each
    clean_answer = answer.replace("---", "").strip()
    # Remove the disclaimer line if it's in the answer (we'll add our own)
    lines = clean_answer.split("\n")
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("⚠️") or "informational purposes only" in stripped.lower():
            continue
        if stripped:
            # Handle bullet points
            if stripped.startswith("- ") or stripped.startswith("• "):
                filtered_lines.append(f"&bull; {stripped[2:]}")
            elif stripped.startswith("* "):
                filtered_lines.append(f"&bull; {stripped[2:]}")
            else:
                filtered_lines.append(stripped)
        else:
            filtered_lines.append("")

    current_para = []
    for line in filtered_lines:
        if line == "":
            if current_para:
                story.append(Paragraph(" ".join(current_para), body_style))
                current_para = []
            story.append(Spacer(1, 4))
        else:
            current_para.append(line)
    if current_para:
        story.append(Paragraph(" ".join(current_para), body_style))

    # Sources
    if citations:
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=8))
        story.append(Paragraph("SOURCES", section_label_style))
        seen = set()
        for c in citations:
            page_str = f"Page {int(c['page']) + 1}" if c.get("page") is not None else "Page N/A"
            key = f"{c['source']}_{page_str}"
            if key not in seen:
                seen.add(key)
                story.append(Paragraph(f"{c['source']} — {page_str}", muted_style))

    # Disclaimer
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=8))
    story.append(Paragraph(
        "This report is for informational purposes only and does not constitute legal advice. "
        "Please consult a qualified legal or compliance professional for guidance specific to your situation.",
        disclaimer_style
    ))

    # Build
    doc.build(story)
    buffer.seek(0)
    return buffer.read()