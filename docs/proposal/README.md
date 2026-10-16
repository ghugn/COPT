# Overleaf proposal: selective neural guidance

Thư mục này là một project Overleaf độc lập cho proposal:

- `main.tex`: nội dung proposal tiếng Việt.
- `references.bib`: tài liệu tham khảo.
- `../../overleaf_proposal_MSO_Lab.zip`: gói mới nhất để upload trực tiếp lên Overleaf.
- `archive/optimal_continuation_v1.tex`: bản ý tưởng cũ, không phải main file.

## Cách dùng trên Overleaf

1. Tạo một Blank Project trên Overleaf.
2. Upload `main.tex` và `references.bib` vào root của project.
3. Chọn compiler **pdfLaTeX**.
4. Recompile hai lần để bibliography và cross-reference được cập nhật đầy đủ.

Bản hiện tại không cần ảnh ngoài. Chưa có bảng kết quả paired COPT; bổ sung sau M1.

## Ranh giới khoa học

Proposal hiện tại nghiên cứu khi nào gọi COPT để hướng dẫn exact search sau khi tính
chi phí inference. Chưa có kết quả xác nhận lợi ích; không dùng số pilot multi-start
cũ để chứng minh giả thuyết mới. Cần paired replay M1 trước khi train gate.

Hướng dẫn benchmark nằm ở `experiments/guidance/README.md` từ root repository.
Archive v1 được giữ để đối chiếu, không đưa vào gói Overleaf hiện hành.
