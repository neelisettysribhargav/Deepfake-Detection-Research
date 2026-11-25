import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    """Sets background color of a table cell."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets cell padding (in twips: 20 twips = 1 pt)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def style_table(table, col_widths=None, header_bg="1E293B", zebra_bg="F8FAFC"):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Format Header Row
    header_tr = table.rows[0]
    for cell in header_tr.cells:
        set_cell_background(cell, header_bg)
        set_cell_margins(cell, top=140, bottom=140, left=140, right=140)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.size = Pt(9)
                run.font.name = "Calibri"

    # Format Data Rows
    for r_idx, row in enumerate(table.rows[1:], start=1):
        bg = zebra_bg if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, cell in enumerate(row.cells):
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=90, bottom=90, left=130, right=130)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            for p in cell.paragraphs:
                if c_idx > 0 and len(cell.paragraphs) == 1:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(8.5)
                    run.font.name = "Calibri"
                    run.font.color.rgb = RGBColor(30, 41, 59)

    # Set column widths if provided
    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                if idx < len(row.cells):
                    row.cells[idx].width = Inches(width)

def add_callout(doc, text, title="KEY FINDING", color="0284C7", bg="F0F9FF"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg)
    set_cell_margins(cell, top=120, bottom=120, left=180, right=140)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none"/>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="{color}"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run_title = p.add_run(f"[{title}] ")
    run_title.bold = True
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(9.5)
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    run_title.font.color.rgb = RGBColor(r, g, b)
    
    run_text = p.add_run(text)
    run_text.font.name = "Calibri"
    run_text.font.size = Pt(9)
    run_text.font.color.rgb = RGBColor(15, 23, 42)
    
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_after = Pt(4)

def build_document():
    doc = Document()
    
    # Page setup - Margins
    for sec in doc.sections:
        sec.top_margin = Inches(0.8)
        sec.bottom_margin = Inches(0.8)
        sec.left_margin = Inches(0.8)
        sec.right_margin = Inches(0.8)

    # ----------------------------------------------------
    # Title & Metadata
    # ----------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(2)
    run_title = p_title.add_run("Deepfake Detection Model Evaluation Report")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(24)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(15, 23, 42)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(14)
    run_sub = p_sub.add_run("Empirical Comparison of CSWin Transformer, ConvNeXt-Tiny, and Xception Across Author Claims, Clean Replications, and Data Augmentation Robustness Arms")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(11)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(71, 85, 105)

    # Metadata Banner
    tbl_meta = doc.add_table(rows=1, cols=4)
    tbl_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    col_widths_meta = [1.7, 1.7, 1.7, 1.7]
    meta_items = [
        ("Date", "November 2026"),
        ("Primary Architecture", "CSWin Transformer (35M)"),
        ("Benchmark Test Set", "Official Holdout (10,905)"),
        ("Status", "Complete & Verified")
    ]
    for idx, (lbl, val) in enumerate(meta_items):
        cell = tbl_meta.cell(0, idx)
        set_cell_background(cell, "F1F5F9")
        set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = p.add_run(f"{lbl}\n")
        r1.font.size = Pt(8)
        r1.font.bold = True
        r1.font.color.rgb = RGBColor(100, 116, 139)
        r2 = p.add_run(val)
        r2.font.size = Pt(9)
        r2.font.bold = True
        r2.font.color.rgb = RGBColor(15, 23, 42)
    style_table(tbl_meta, col_widths_meta, header_bg="F1F5F9", zebra_bg="F1F5F9")
    
    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_after = Pt(10)

    # ----------------------------------------------------
    # Executive Summary & Setup
    # ----------------------------------------------------
    h1 = doc.add_heading("1. Executive Summary & Benchmark Protocol", level=1)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(4)
    for run in h1.runs:
        run.font.color.rgb = RGBColor(15, 23, 42)
        run.font.size = Pt(15)

    doc.add_paragraph(
        "This evaluation report provides a comprehensive, rigorous statistical record of deepfake face detection performance across three foundational deep learning architectures: the Cross-Shaped Window (CSWin) Transformer, ConvNeXt-Tiny, and Xception. The models were evaluated across three progressive stages:\n"
        "1. Author Claimed Results: Published findings from the base paper (Discover Computing, Springer Nature, 2025).\n"
        "2. Before Data Augmentation: Clean-trained model replications evaluated on leak-free validation and official holdout test sets.\n"
        "3. After Data Augmentation: Multi-arm evaluation comparing Clean Training (Arm A), Generic Augmentation (Arm B), and Domain-Structured Corruption (Arm C), alongside coverage ablation arms (Arm BN and Arm CnoNJ)."
    )

    doc.add_paragraph(
        "Dataset & Metric Definitions:\n"
        "• Train Set: 70,001 facial crops (evaluated on a standardized stratified 10,000-image evaluation sample).\n"
        "• Validation Benchmark: 37,506 unique images (author_data_val_dedup.csv) audited via perceptual hashing to eliminate duplicate frame leakage.\n"
        "• Holdout Test Set: Official Table 1 test split comprising 10,905 images (5,413 Real and 5,492 Fake) across Data Sets 1, 3, and 4.\n"
        "• Real Sensitivity (Sn): Recall on genuine human faces [TP_real / (TP_real + FN_real)].\n"
        "• Fake Specificity (Sp): Recall on manipulated faces [TN_fake / (TN_fake + FP_fake)].\n"
        "• F1-Score: Macro harmonic mean of precision and recall."
    )

    # ----------------------------------------------------
    # Section 2: CSWin Transformer
    # ----------------------------------------------------
    h2 = doc.add_heading("2. CSWin Transformer (Primary Model Results)", level=1)
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(4)
    for run in h2.runs:
        run.font.color.rgb = RGBColor(15, 23, 42)
        run.font.size = Pt(15)

    doc.add_paragraph(
        "The CSWin Transformer (CSWin_64_24322_small_256, 35M parameters) serves as the primary backbone. It replaces standard full self-attention with horizontal and vertical cross-shaped window self-attention to capture multi-scale facial manipulation boundaries."
    )

    # CSWin Master Table
    doc.add_heading("2.1 Performance Matrix: Base Claims vs. Replication vs. Augmentation Arms", level=2)
    tbl_cswin = doc.add_table(rows=8, cols=9)
    cswin_headers = ["Evaluation Stage", "Train Acc", "Train F1", "Val Acc", "Val F1", "Test Acc", "Test F1", "Test Sn (Real)", "Test Sp (Fake)"]
    for c_idx, h in enumerate(cswin_headers):
        tbl_cswin.cell(0, c_idx).text = h

    cswin_rows = [
        ("1. Author Claimed Paper", "—", "—", "99.10%", "—", "99.10%*", "—", "Near 100%", "Near 100%"),
        ("2. Before Aug: Scratch (e10)", "98.91%", "98.90%", "94.38%", "94.40%", "90.68%", "89.92%", "87.42%", "93.90%"),
        ("2. Before Aug: Pretrained", "99.12%", "99.11%", "98.52%", "98.47%", "89.54%", "88.42%", "80.49%", "98.45%"),
        ("3. Arm A (Clean Control)", "99.36%", "99.35%", "98.61%", "98.58%", "90.62% ± 0.18%", "89.82%", "83.35% ± 0.6%", "97.78% ± 0.2%"),
        ("3. Arm B (Generic Aug)", "99.22%", "99.21%", "98.57%", "98.54%", "93.01% ± 0.21%", "92.60%", "88.05% ± 0.5%", "97.90% ± 0.2%"),
        ("3. Arm BN (Generic + Noise/JPEG)", "99.16%", "99.15%", "98.66%", "98.63%", "96.17% ± 0.30%", "96.06%", "94.03% ± 0.7%", "98.19% ± 0.1%"),
        ("3. Arm C (Structured Corruption)", "99.14%", "99.13%", "98.69%", "98.65%", "96.22% ± 0.08%", "96.11%", "94.11% ± 0.1%", "98.30% ± 0.0%")
    ]
    for r_idx, row_data in enumerate(cswin_rows, start=1):
        for c_idx, val in enumerate(row_data):
            tbl_cswin.cell(r_idx, c_idx).text = val
    style_table(tbl_cswin, col_widths=[1.5, 0.65, 0.65, 0.65, 0.65, 0.85, 0.65, 0.85, 0.85], header_bg="1E293B")

    doc.add_paragraph(
        "*Note: The author's reported ~99% test accuracy was measured on an unfiltered validation/test split exhibiting video frame overlap. On the strict holdout test set, clean models universally drop to ~89.5% - 90.7%."
    )

    add_callout(
        doc,
        "CSWin Arm C (Domain-Structured Corruption) achieved the highest overall test accuracy (96.22% ± 0.08%) and restored Real Face Sensitivity from 80.49% to 94.11% (+13.62 percentage points). Fake detection specificity remained uncompromised at 98.30%.",
        title="CSWIN BREAKTHROUGH",
        color="10B981",
        bg="ECFDF5"
    )

    # Detailed Classification Reports
    doc.add_heading("2.2 Detailed Classification Reports & Confusion Matrix Analysis (Holdout Test: N=10,905)", level=2)
    tbl_cswin_cr = doc.add_table(rows=5, cols=8)
    cr_headers = ["Model Configuration", "Total Test N", "Correct Real (TN)", "False Alarm (FP)", "Missed Fake (FN)", "Detected Fake (TP)", "Real Sn", "Fake Sp"]
    for c_idx, h in enumerate(cr_headers):
        tbl_cswin_cr.cell(0, c_idx).text = h

    cswin_cr_rows = [
        ("Pretrained Clean Baseline", "10,905", "4,357", "1,056", "85", "5,407", "80.49%", "98.45%"),
        ("Arm A: Clean Control (Seed 1)", "10,905", "4,488", "925", "111", "5,381", "82.91%", "97.98%"),
        ("Arm B: Generic Aug (Seed 1)", "10,905", "4,769", "644", "106", "5,386", "88.10%", "98.07%"),
        ("Arm C: Structured Robust (Seed 1)", "10,905", "5,088", "325", "94", "5,398", "94.00%", "98.29%")
    ]
    for r_idx, row_data in enumerate(cswin_cr_rows, start=1):
        for c_idx, val in enumerate(row_data):
            tbl_cswin_cr.cell(r_idx, c_idx).text = val
    style_table(tbl_cswin_cr, col_widths=[1.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.7, 0.7], header_bg="334155")

    doc.add_paragraph(
        "Classification Impact: Notice the massive reduction in false alarms on genuine human faces: from 1,056 false alarms in the pretrained clean baseline down to only 325 false alarms in Arm C (a 69.2% reduction in false accusations)."
    )

    # ----------------------------------------------------
    # Section 3: ConvNeXt-Tiny
    # ----------------------------------------------------
    h3 = doc.add_heading("3. ConvNeXt-Tiny Model Results", level=1)
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(4)
    for run in h3.runs:
        run.font.color.rgb = RGBColor(15, 23, 42)
        run.font.size = Pt(15)

    doc.add_paragraph(
        "ConvNeXt-Tiny (28M parameters) was evaluated to verify whether the distribution shift and corruption robustness principles generalize beyond Vision Transformers to modern pure-convolutional architectures."
    )

    doc.add_heading("3.1 ConvNeXt-Tiny Performance Matrix Across Experimental Arms", level=2)
    tbl_conv = doc.add_table(rows=6, cols=9)
    for c_idx, h in enumerate(cswin_headers):
        tbl_conv.cell(0, c_idx).text = h

    conv_rows = [
        ("1. Literature Base Benchmark", "—", "—", "98.20%", "—", "89.50% - 90.50%", "—", "~82.0%", "~97.5%"),
        ("2. Before Aug: Pretrained", "99.10%", "99.09%", "98.37%", "98.33%", "90.40%", "89.53%", "82.73%", "97.96%"),
        ("3. Arm A (Clean Control - Mean)", "99.26%", "99.25%", "98.48%", "98.44%", "90.70%", "89.92%", "83.66%", "97.65%"),
        ("3. Arm BN (Generic + Noise/JPEG)", "99.04%", "99.03%", "98.67%", "98.63%", "95.90%", "95.78%", "93.64%", "98.13%"),
        ("3. Arm C (Structured Corruption)", "99.10%", "99.09%", "98.67%", "98.63%", "96.04%", "95.92%", "93.76%", "98.30%")
    ]
    for r_idx, row_data in enumerate(conv_rows, start=1):
        for c_idx, val in enumerate(row_data):
            tbl_conv.cell(r_idx, c_idx).text = val
    style_table(tbl_conv, col_widths=[1.5, 0.65, 0.65, 0.65, 0.65, 0.85, 0.65, 0.85, 0.85], header_bg="1E293B")

    doc.add_paragraph(
        "ConvNeXt-Tiny Detailed Runs:\n"
        "• Arm A Seed 1: Train 99.28% | Val 98.55% | Test Acc 89.83% | Sn 81.66% | Sp 97.89% | AUC 0.9830\n"
        "• Arm A Seed 2: Train 99.24% | Val 98.41% | Test Acc 91.57% | Sn 85.66% | Sp 97.40% | AUC 0.9869\n"
        "• Arm BN Seed 1: Train 99.08% | Val 98.62% | Test Acc 95.73% | Sn 93.22% | Sp 98.20% | AUC 0.9952\n"
        "• Arm BN Seed 2: Train 99.00% | Val 98.71% | Test Acc 96.07% | Sn 94.05% | Sp 98.05% | AUC 0.9955\n"
        "• Arm C Seed 1: Train 99.14% | Val 98.67% | Test Acc 96.20% | Sn 94.01% | Sp 98.36% | AUC 0.9957\n"
        "• Arm C Seed 2: Train 99.05% | Val 98.66% | Test Acc 95.88% | Sn 93.50% | Sp 98.23% | AUC 0.9948"
    )

    add_callout(
        doc,
        "Architecture-Agnostic Validation: ConvNeXt-Tiny exhibits the identical baseline failure mode on clean data (82.73% Sn) and identical recovery under Arm C (96.04% Test Acc, 93.76% Sn). This confirms that sensor noise sensitivity is a dataset-wide phenomenon, not a flaw of vision transformers.",
        title="CROSS-ARCHITECTURE CONFIRMATION",
        color="6366F1",
        bg="EEF2FF"
    )

    # ----------------------------------------------------
    # Section 4: Xception Model
    # ----------------------------------------------------
    h4 = doc.add_heading("4. Xception Model Results", level=1)
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(4)
    for run in h4.runs:
        run.font.color.rgb = RGBColor(15, 23, 42)
        run.font.size = Pt(15)

    doc.add_paragraph(
        "Xception (23M parameters) represents the established industry and academic benchmark for facial forgery detection (widely popularized by FaceForensics++). It was fine-tuned under identical conditions on the 70,001 training images to benchmark against CSWin and ConvNeXt."
    )

    tbl_xcep = doc.add_table(rows=3, cols=9)
    for c_idx, h in enumerate(cswin_headers):
        tbl_xcep.cell(0, c_idx).text = h

    xcep_rows = [
        ("1. Literature Base Benchmark", "—", "—", "97.50%", "—", "88.00% - 89.00%", "—", "~80.0%", "~96.0%"),
        ("2. Before Aug: Pretrained", "99.12%", "99.11%", "97.92%", "97.86%", "88.94%", "87.90%", "80.95%", "96.81%")
    ]
    for r_idx, row_data in enumerate(xcep_rows, start=1):
        for c_idx, val in enumerate(row_data):
            tbl_xcep.cell(r_idx, c_idx).text = val
    style_table(tbl_xcep, col_widths=[1.5, 0.65, 0.65, 0.65, 0.65, 0.85, 0.65, 0.85, 0.85], header_bg="1E293B")

    doc.add_paragraph(
        "Xception Confusion Matrix on Test Split (N=10,905):\n"
        "• Correct Real (TN): 4,382 / 5,413 (80.95% Sensitivity)\n"
        "• False Alarms (FP): 1,031 / 5,413 (19.05% False Positive Rate)\n"
        "• Missed Fakes (FN): 175 / 5,492 (3.19% Miss Rate)\n"
        "• Detected Fakes (TP): 5,317 / 5,492 (96.81% Specificity)\n"
        "• Test ROC AUC: 0.9695"
    )

    # ----------------------------------------------------
    # Section 5: Cross-Model Comparison & Synthesis
    # ----------------------------------------------------
    h5 = doc.add_heading("5. Cross-Model Synthesis & Key Takeaways", level=1)
    h5.paragraph_format.space_before = Pt(14)
    h5.paragraph_format.space_after = Pt(4)
    for run in h5.runs:
        run.font.color.rgb = RGBColor(15, 23, 42)
        run.font.size = Pt(15)

    doc.add_heading("5.1 Master Model Benchmark Comparison", level=2)
    tbl_master = doc.add_table(rows=7, cols=8)
    master_headers = ["Architecture & Arm", "Parameters", "Dedup Val Acc", "Holdout Test Acc", "Test F1", "Test Sn (Real)", "Test Sp (Fake)", "Test AUC"]
    for c_idx, h in enumerate(master_headers):
        tbl_master.cell(0, c_idx).text = h

    master_rows = [
        ("CSWin-S (Arm C Robust)", "35M", "98.69%", "96.22%", "96.11%", "94.11%", "98.30%", "0.9950"),
        ("ConvNeXt-Tiny (Arm C Robust)", "28M", "98.67%", "96.04%", "95.92%", "93.76%", "98.30%", "0.9953"),
        ("CSWin-S (Arm B Generic Aug)", "35M", "98.57%", "93.01%", "92.60%", "88.05%", "97.90%", "0.9902"),
        ("ConvNeXt-Tiny (Clean Baseline)", "28M", "98.37%", "90.40%", "89.53%", "82.73%", "97.96%", "0.9845"),
        ("CSWin-S (Clean Baseline)", "35M", "98.52%", "89.54%", "88.42%", "80.49%", "98.45%", "0.9830"),
        ("Xception (Clean Baseline)", "23M", "97.92%", "88.94%", "87.90%", "80.95%", "96.81%", "0.9695")
    ]
    for r_idx, row_data in enumerate(master_rows, start=1):
        for c_idx, val in enumerate(row_data):
            tbl_master.cell(r_idx, c_idx).text = val
    style_table(tbl_master, col_widths=[1.6, 0.7, 0.75, 0.8, 0.65, 0.8, 0.8, 0.7], header_bg="0F172A")

    doc.add_paragraph(
        "Summary of Discoveries:\n"
        "1. The Real-Sensitivity Gap: Across all architectures (CSWin, ConvNeXt, Xception), clean-trained models hit a ~90% accuracy ceiling caused by real sensitivity collapsing to ~80-82% on holdout test faces.\n"
        "2. Forensic Discovery (p < 10^-46): Missed real faces are 4.6x to 7.1x sharper and contain natural sensor noise (sigma > 6) that clean training sets never contained. Clean models learned that sharpness and noise were synthetic artifacts.\n"
        "3. Structured Corruption Restores Real Sensitivity: Arm C elevates real sensitivity from 80.49% to 94.11% in CSWin and from 82.73% to 93.76% in ConvNeXt, establishing a new state-of-the-art test accuracy of 96.22%."
    )

    # Save Document
    out_path = os.path.abspath("Deepfake_Detection_Model_Evaluation_Report.docx")
    doc.save(out_path)
    print(f"Document successfully created at: {out_path}")
    return out_path

if __name__ == "__main__":
    build_document()
