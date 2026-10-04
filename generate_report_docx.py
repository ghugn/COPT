import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pathlib import Path

def create_concise_report():
    doc = Document()

    # Margins
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    NAVY = RGBColor(24, 43, 73)
    CHARCOAL = RGBColor(34, 34, 34)

    # Title
    p_title = doc.add_paragraph()
    r_title = p_title.add_run("TÓM TẮT Ý NGHĨA 4 BẢNG BENCHMARK (COPT)")
    r_title.font.name = "Arial"
    r_title.font.size = Pt(16)
    r_title.font.bold = True
    r_title.font.color.rgb = NAVY
    p_title.paragraph_format.space_after = Pt(16)

    # Content lines
    items = [
        ("Table 3 (MIS ↔ MVC): ", "Khảo sát chuyển giao giữa 2 bài toán bù nhau giữ nguyên cấu trúc đồ thị (chứng minh fine-tune đảo trọng số đầu ra vượt qua cả train từ đầu)."),
        ("Table 4 (MIS → MaxClique): ", "Khảo sát chuyển giao qua đồ thị bù (chứng minh giải MIS trên đồ thị bù Ḡ cho kết quả giải MaxClique tối ưu nhất)."),
        ("Table 7 (Multi-Task): ", "Huấn luyện mô hình nền tảng đa nhiệm trên cả 6 bài toán NP-hard (chứng minh fine-tune từ mô hình nền tảng vượt trội so với mô hình đơn nhiệm)."),
        ("Table 5 (Leave-One-Out): ", "Đánh giá khả năng chuyển giao siêu tốc trong điều kiện ít tài nguyên (chỉ 20 epochs từ 5 bài toán sang bài toán còn lại)."),
    ]

    for title, desc in items:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.line_spacing = 1.2
        r_b = p.add_run(title)
        r_b.font.name = "Arial"
        r_b.font.size = Pt(11)
        r_b.font.bold = True
        r_b.font.color.rgb = CHARCOAL
        
        r_d = p.add_run(desc)
        r_d.font.name = "Arial"
        r_d.font.size = Pt(11)
        r_d.font.color.rgb = CHARCOAL

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Ghi chú 1
    p_note1 = doc.add_paragraph()
    p_note1.paragraph_format.space_after = Pt(8)
    p_note1.paragraph_format.line_spacing = 1.2
    r_n1_title = p_note1.add_run("Ghi chú thực nghiệm: ")
    r_n1_title.font.name = "Arial"
    r_n1_title.font.size = Pt(11)
    r_n1_title.font.bold = True
    r_n1_title.font.color.rgb = NAVY

    r_n1_desc = p_note1.add_run("Các mốc Baseline train từ đầu (from scratch) được lấy trực tiếp số liệu từ bài báo làm mốc đối chiếu để tránh lãng phí hàng chục giờ GPU, dồn toàn bộ tài nguyên kiểm chứng khả năng học chuyển giao.")
    r_n1_desc.font.name = "Arial"
    r_n1_desc.font.size = Pt(11)
    r_n1_desc.font.color.rgb = CHARCOAL

    # Ghi chú 2 (đúng dòng yêu cầu về Table 4)
    p_note2 = doc.add_paragraph()
    p_note2.paragraph_format.space_after = Pt(8)
    p_note2.paragraph_format.line_spacing = 1.2
    r_n2_title = p_note2.add_run("Về Table 4: ")
    r_n2_title.font.name = "Arial"
    r_n2_title.font.size = Pt(11)
    r_n2_title.font.bold = True
    r_n2_title.font.color.rgb = NAVY

    r_n2_desc = p_note2.add_run("Vì Table 4 chạy quá nhiều, tốn nhiều thời gian và không quan trọng bằng các bảng mô hình nền tảng ở trên nên không chạy thực nghiệm Table 4 mà sử dụng luôn số liệu của paper.")
    r_n2_desc.font.name = "Arial"
    r_n2_desc.font.size = Pt(11)
    r_n2_desc.font.color.rgb = CHARCOAL

    paths = [
        Path("d:/Downloads/Lab_resource/Task5-MHoang/COPT-MT-main/Bao_Cao_Y_Nghia_Benchmark_COPT.docx"),
        Path("d:/Downloads/Lab_resource/Task5-MHoang/COPT-MT-main/Giai_thich_Bench.docx"),
    ]
    for p in paths:
        doc.save(p)
        print(f"Saved: {p}")

if __name__ == "__main__":
    create_concise_report()
