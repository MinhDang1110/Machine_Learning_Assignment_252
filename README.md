# Bài tập lớn Học Máy - CO3117

## Thông tin môn học

| Mục | Thông tin |
|---|---|
| Tên môn học | Học Máy |
| Mã môn học | CO3117 |
| Học kỳ | HK252 |
| Năm học | 2025 - 2026 |
| Giảng viên hướng dẫn | Trương Vĩnh Lân |
| Nhóm thực hiện | Nhóm 8 |

---

## Thông tin thành viên nhóm

| STT | Họ và tên | MSSV | Email |
|---:|---|---|---|
| 1 | Đặng Đình Minh Hoàng | 2411068 | hoang.dangdinhminh@hcmut.edu.vn |
| 2 | Nguyễn Ngọc Thành Đạt | 2111013 | Chưa cung cấp |
| 3 | Phạm Minh Đức | 2414106 | Chưa cung cấp |
| 4 | Ngô Đức Thắng | 2313184 | Chưa cung cấp |
| 5 | Nguyễn Hữu Khánh | 2211521 | Chưa cung cấp |

---

## Mục tiêu của bài tập lớn

Bài tập lớn nhằm xây dựng và đánh giá một hệ thống học máy hoàn chỉnh cho bài toán phân loại văn bản đa lớp trên bộ dữ liệu **AG News Classification Dataset**. Bộ dữ liệu gồm bốn chủ đề chính:

- World
- Sports
- Business
- Sci/Tech

Nhóm triển khai và so sánh hai hướng tiếp cận chính:

1. **Pipeline học máy truyền thống**
   - Tiền xử lý văn bản.
   - Trích xuất đặc trưng bằng Bag of Words và TF-IDF.
   - Huấn luyện các mô hình Naive Bayes, Logistic Regression và Linear SVM.
   - Đánh giá bằng Accuracy, Precision, Recall, Macro F1 và Weighted F1.

2. **Pipeline học sâu**
   - Sử dụng DistilBERT để trích xuất embedding ngữ nghĩa cho văn bản.
   - Lưu đặc trưng dưới định dạng `.npy`.
   - Huấn luyện mô hình MLP Neural Network trên đặc trưng DistilBERT.
   - So sánh kết quả với pipeline truyền thống.

Mục tiêu cuối cùng là phân tích hiệu quả của các phương pháp biểu diễn văn bản khác nhau và đánh giá sự khác biệt giữa pipeline học máy truyền thống và pipeline học sâu.

---

## Cấu trúc thư mục dự án

```text
.
├── modules/
│   ├── models.py
│   ├── feature_extraction.py
│   ├── preprocessing.py
│   └── utils.py
│
├── notebooks/
│   └── ML_pipeline.ipynb
│
├── features/
│   ├── .gitattributes
│   ├── ft.zip
│   └── ft_deeplearning.zip
│
├── reports/
│   └── ML_Report_252.pdf
│
└── README.md
```

### Mô tả các thư mục chính

| Thư mục / File | Mô tả |
|---|---|
| `modules/` | Chứa các module Python hỗ trợ tiền xử lý, trích xuất đặc trưng, định nghĩa mô hình và các hàm tiện ích. |
| `notebooks/ML_pipeline.ipynb` | Notebook chính dùng để chạy toàn bộ pipeline trên Google Colab. |
| `features/ft.zip` | File nén chứa đặc trưng đã trích xuất cho pipeline truyền thống. |
| `features/ft_deeplearning.zip` | File nén chứa đặc trưng DistilBERT cho pipeline học sâu. |
| `reports/ML_Report_252.pdf` | Báo cáo PDF của bài tập lớn. |
| `README.md` | File mô tả thông tin dự án, cách chạy và cấu trúc mã nguồn. |

---

## Hướng dẫn chạy notebook

### Cách 1: Chạy trực tiếp trên Google Colab

Mở notebook tại đường dẫn:

[Google Colab Notebook](https://colab.research.google.com/drive/1rfqEpD1Tes7E8S8sCEH5n2c425wsI88m?usp=sharing)

Sau đó chọn:

```text
Runtime → Run all
```

Notebook được thiết kế để tự động thực hiện các bước chính:

1. Cài đặt hoặc import các thư viện cần thiết.
2. Tải dữ liệu từ nguồn công khai.
3. Tải các file đặc trưng đã trích xuất từ Hugging Face.
4. Giải nén đặc trưng vào thư mục `features/`.
5. Huấn luyện và đánh giá các mô hình truyền thống.
6. Huấn luyện và đánh giá mô hình học sâu.
7. Xuất kết quả so sánh ra thư mục `reports/`.

### Cách 2: Chạy từ repository GitHub

Clone repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Mở notebook:

```bash
notebooks/ML_pipeline.ipynb
```

Sau đó chạy notebook bằng Google Colab hoặc Jupyter Notebook.

---

## Yêu cầu thư viện

Các thư viện chính được sử dụng trong dự án gồm:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
torch
transformers
tqdm
scipy
```

Nếu chạy trên Google Colab, các thư viện phổ biến như `numpy`, `pandas`, `matplotlib`, `scikit-learn` và `torch` thường đã có sẵn. Một số thư viện bổ sung sẽ được cài đặt trực tiếp trong notebook nếu cần.

---

## Dữ liệu và file đặc trưng

### Dataset

Dự án sử dụng bộ dữ liệu **AG News Classification Dataset** cho bài toán phân loại văn bản đa lớp. Dataset được tải trực tiếp trong notebook từ nguồn công khai, không phụ thuộc vào Google Drive hoặc Dropbox cá nhân.

### File đặc trưng pipeline truyền thống

File đặc trưng cho pipeline truyền thống gồm các vector Bag of Words và TF-IDF đã được trích xuất sẵn.

Link tải:

[ft.zip](https://huggingface.co/datasets/hoangminh1110/ML_Assignment252_Feature/resolve/main/ft.zip)

### File đặc trưng pipeline học sâu

File đặc trưng cho pipeline học sâu gồm các embedding DistilBERT đã được trích xuất sẵn.

Link tải:

[ft_deeplearning.zip](https://huggingface.co/datasets/hoangminh1110/ML_Assignment252_Feature/resolve/main/ft_deeplearning.zip)

---

## Kết quả chính

Kết quả thực nghiệm cho thấy:

| Pipeline | Mô hình tốt nhất | Đặc trưng | Accuracy | Macro F1 |
|---|---|---|---:|---:|
| Truyền thống | Linear SVM | Bag of Words | 0.9087 | 0.9085 |
| Học sâu | MLP Neural Network | DistilBERT Embedding | 0.9142 | 0.9142 |

Pipeline học sâu sử dụng **DistilBERT Embedding + MLP Neural Network** đạt kết quả tốt nhất, nhưng có chi phí huấn luyện cao hơn. Pipeline truyền thống với **Linear SVM + Bag of Words** vẫn đạt kết quả cạnh tranh và có thời gian huấn luyện thấp hơn đáng kể.

---

## Báo cáo và liên kết

| Nội dung | Đường dẫn |
|---|---|
| Google Colab Notebook | [ML_pipeline.ipynb](https://colab.research.google.com/drive/1rfqEpD1Tes7E8S8sCEH5n2c425wsI88m?usp=sharing) |
| Đặc trưng truyền thống | [ft.zip](https://huggingface.co/datasets/hoangminh1110/ML_Assignment252_Feature/resolve/main/ft.zip) |
| Đặc trưng học sâu | [ft_deeplearning.zip](https://huggingface.co/datasets/hoangminh1110/ML_Assignment252_Feature/resolve/main/ft_deeplearning.zip) |

---

## Ghi chú

- Notebook không mount Google Drive hoặc Dropbox cá nhân.
- Dữ liệu và đặc trưng được tải từ nguồn công khai.
- Các file đặc trưng được lưu dưới định dạng `.npy` sau khi giải nén.
- Kết quả thực nghiệm được lưu trong thư mục `reports/`.
- Nếu chạy lại toàn bộ notebook, nên sử dụng môi trường Google Colab có RAM đủ lớn để xử lý đặc trưng văn bản.
