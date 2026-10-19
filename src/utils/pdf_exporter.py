"""
PDF and JSON Audit Report Exporter using ReportLab.
Generates enterprise-grade clinical trial matching and safety audit reports.
"""
import io
import json
from datetime import datetime, timezone
from typing import Dict, Any, List

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


def export_audit_json(state: Dict[str, Any]) -> str:
    """
    Exports complete audit state as formatted JSON string.
    """
    export_payload = {
        "metadata": {
            "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "generator": "Autonomous Clinical Trial Matching & Protocol Deviation Auditor v1.0",
            "patient_id": state.get("patient_id", "UNKNOWN"),
            "protocol_id": state.get("protocol", {}).get("protocol_id", "UNKNOWN")
        },
        "verdict": {
            "critic_approved": state.get("critic_approved", False),
            "eligibility_score": state.get("eligibility_score", 0.0),
            "status": "APPROVED" if state.get("critic_approved") else "REJECTED"
        },
        "gemini_clinical_safety_review": state.get("llm_safety_narrative", ""),
        "safety_and_deviations": {
            "safety_alerts": state.get("safety_alerts", []),
            "deviation_warnings": state.get("deviation_warnings", [])
        },
        "rule_match_results": state.get("match_results", []),
        "normalized_ehr": state.get("normalized_ehr", {}),
        "protocol": state.get("protocol", {}),
        "audit_logs": state.get("audit_logs", [])
    }
    return json.dumps(export_payload, indent=2, default=str)


def generate_audit_pdf(state: Dict[str, Any]) -> bytes:
    """
    Generates a professional, publication-quality PDF audit report from state.
    Returns the generated PDF as raw bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F2942'),
        alignment=TA_LEFT
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4A5568'),
        alignment=TA_LEFT
    )

    section_header = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=8,
        spaceAfter=4
    )

    cell_header = ParagraphStyle(
        'CellHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=TA_LEFT
    )

    cell_body = ParagraphStyle(
        'CellBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#1F2937')
    )

    pass_style = ParagraphStyle(
        'PassBadge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#15803D')
    )

    fail_style = ParagraphStyle(
        'FailBadge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#B91C1C')
    )

    alert_style = ParagraphStyle(
        'AlertText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#7F1D1D')
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("CLINICAL TRIAL ELIGIBILITY & SAFETY AUDIT REPORT", title_style))
    elements.append(Paragraph(
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} | "
        f"System: Autonomous Clinical Trial Auditor (LangGraph Agentic Pipeline)",
        subtitle_style
    ))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=12))

    # 2. Executive Summary Metrics Table
    patient_id = state.get("patient_id", "UNKNOWN")
    protocol = state.get("protocol", {})
    protocol_id = protocol.get("protocol_id", "UNKNOWN")
    protocol_title = protocol.get("title", "Clinical Protocol")
    score = state.get("eligibility_score", 0.0)
    approved = state.get("critic_approved", False)

    verdict_text = "APPROVED (ELIGIBLE)" if approved else "REJECTED (INELIGIBLE)"
    verdict_color = colors.HexColor('#DCFCE7') if approved else colors.HexColor('#FEE2E2')
    verdict_text_color = colors.HexColor('#15803D') if approved else colors.HexColor('#B91C1C')

    ehr = state.get("normalized_ehr", {})
    age_display = ehr.get("age", "N/A")
    egfr_display = f"{ehr.get('egfr', 0):.1f} mL/min/1.73m²"
    platelets_display = f"{ehr.get('platelets', 0):,.0f} /µL"
    hba1c_display = f"{ehr.get('hba1c', 0):.1f}%"

    summary_data = [
        [
            Paragraph("<b>Patient Identifier:</b>", cell_body),
            Paragraph(str(patient_id), cell_body),
            Paragraph("<b>Protocol ID:</b>", cell_body),
            Paragraph(f"{protocol_id}", cell_body)
        ],
        [
            Paragraph("<b>Patient Demographics:</b>", cell_body),
            Paragraph(f"Age: {age_display} | Gender: {ehr.get('gender', 'N/A')}", cell_body),
            Paragraph("<b>Trial Title:</b>", cell_body),
            Paragraph(protocol_title, cell_body)
        ],
        [
            Paragraph("<b>Renal / Platelet Labs:</b>", cell_body),
            Paragraph(f"eGFR: {egfr_display} | Plt: {platelets_display}", cell_body),
            Paragraph("<b>Eligibility Score:</b>", cell_body),
            Paragraph(f"<b>{score:.1f}%</b>", cell_body)
        ],
        [
            Paragraph("<b>Metabolic Labs:</b>", cell_body),
            Paragraph(f"HbA1c: {hba1c_display} | ECOG: {ehr.get('ecog', 0)}", cell_body),
            Paragraph("<b>Critic Gatekeeper Verdict:</b>", cell_body),
            Paragraph(f"<b>{verdict_text}</b>", ParagraphStyle('V', fontName='Helvetica-Bold', fontSize=9, textColor=verdict_text_color))
        ]
    ]

    summary_table = Table(summary_data, colWidths=[110, 160, 110, 160])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 12))

    # 3. Safety Alerts and Protocol Deviations Section
    safety_alerts = state.get("safety_alerts", [])
    deviation_warnings = state.get("deviation_warnings", [])

    if safety_alerts or deviation_warnings:
        elements.append(Paragraph("CRITICAL SAFETY ALERTS & PROTOCOL DEVIATIONS", section_header))
        alert_rows = []
        for sa in safety_alerts:
            alert_rows.append([
                Paragraph("<b>SAFETY CONTRAINDICATION:</b>", alert_style),
                Paragraph(sa, alert_style)
            ])
        for dw in deviation_warnings:
            alert_rows.append([
                Paragraph("<b>DEVIATION WARNING:</b>", alert_style),
                Paragraph(dw, alert_style)
            ])

        alert_table = Table(alert_rows, colWidths=[150, 390])
        alert_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FEF2F2')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#FCA5A5')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#FECACA')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(alert_table)
        elements.append(Spacer(1, 12))

    # 4. Gemini AI Clinical Safety Review Narrative
    llm_narrative = state.get("llm_safety_narrative", "").strip()
    if llm_narrative and "Deterministic safety audit completed" not in llm_narrative:
        elements.append(Paragraph("GEMINI AI MEDICAL MONITOR & SAFETY REVIEW", section_header))
        formatted_narrative = llm_narrative.replace("\n", "<br/>")
        narrative_para = Paragraph(
            f"<font size='8' color='#1E293B'>{formatted_narrative}</font>",
            ParagraphStyle('GeminiNarrative', fontName='Helvetica', fontSize=8, leading=11)
        )
        narrative_table = Table([[narrative_para]], colWidths=[540])
        narrative_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0F9FF')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#38BDF8')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(narrative_table)
        elements.append(Spacer(1, 12))

    # 5. Detailed Rule Matching Matrix
    elements.append(Paragraph("DETERMINISTIC PROTOCOL RULE AUDIT MATRIX", section_header))

    matrix_headers = [
        Paragraph("<b>Rule ID</b>", cell_header),
        Paragraph("<b>Type</b>", cell_header),
        Paragraph("<b>Criterion Description</b>", cell_header),
        Paragraph("<b>Target</b>", cell_header),
        Paragraph("<b>Observed</b>", cell_header),
        Paragraph("<b>Status</b>", cell_header)
    ]
    matrix_rows = [matrix_headers]

    match_results = state.get("match_results", [])
    for res in match_results:
        status_para = Paragraph("<b>PASS</b>", pass_style) if res.get("passed") else Paragraph("<b>FAIL</b>", fail_style)
        criterion_type = res.get("criterion_type", "INC")
        target_display = f"{res.get('operator', '')} {res.get('target_value', '')} {res.get('unit', '')}".strip()
        actual_display = str(res.get("actual_value", "N/A"))

        matrix_rows.append([
            Paragraph(res.get("rule_id", ""), cell_body),
            Paragraph(criterion_type, cell_body),
            Paragraph(res.get("description", ""), cell_body),
            Paragraph(target_display, cell_body),
            Paragraph(actual_display, cell_body),
            status_para
        ])

    matrix_table = Table(matrix_rows, colWidths=[55, 60, 220, 85, 75, 45])
    matrix_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    elements.append(matrix_table)
    elements.append(Spacer(1, 14))

    # 5. System Execution Audit Log Excerpt
    elements.append(Paragraph("TRACEABILITY & AUDIT TRAIL SUMMARY", section_header))
    logs = state.get("audit_logs", [])
    log_text = "<br/>".join(logs[-10:]) if logs else "No execution logs recorded."
    log_para = Paragraph(
        f"<font size='6' color='#475569'>{log_text}</font>",
        ParagraphStyle('Logs', fontName='Helvetica', fontSize=6, leading=9)
    )
    log_table = Table([[log_para]], colWidths=[540])
    log_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F5F9')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#94A3B8')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(log_table)

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
