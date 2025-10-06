# Weather-type-prediction-on-tabular-dataset

Đây là dự án học máy nhằm mục đích phân loại và dự đoán loại hình thời tiết dựa trên một bộ dữ liệu dạng bảng. Dự án tập trung vào việc áp dụng và so sánh hiệu suất của hai thuật toán mạnh mẽ: **Support Vector Machine (SVM)** và **XGBoost**.

## Mục lục
1. [Mục tiêu dự án](#mục-tiêu-dự-án)
2. [Mô tả dữ liệu](#mô-tả-dữ-liệu)
3. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
4. [Phân công nhiệm vụ](#phân-công-nhiệm-vụ)
5. [Yêu cầu Báo cáo (Report)](#yêu-cầu-báo-cáo-report)
6. [Deadline](#deadline)

## Mục tiêu dự án
- **Tiền xử lý và chuẩn bị dữ liệu:** Làm sạch, xử lý các giá trị thiếu, mã hóa các biến categorical, và chuẩn hóa dữ liệu để phù hợp với các mô hình.
- **Xây dựng mô hình SVM:** Áp dụng thuật toán Support Vector Machine để phân loại thời tiết. Tinh chỉnh các tham số (hyperparameter tuning) để đạt hiệu suất tốt nhất.
- **Xây dựng mô hình XGBoost:** Áp dụng thuật toán Extreme Gradient Boosting, một thuật toán mạnh mẽ dựa trên cây quyết định, để phân loại thời tiết và tinh chỉnh tham số.
- **Đánh giá hiệu suất:** Sử dụng các độ đo (metrics) phổ biến như Accuracy, Precision, Recall, F1-score và Confusion Matrix để đánh giá hiệu suất của cả hai mô hình.
- **So sánh và kết luận:** So sánh ưu, nhược điểm và kết quả của hai phương pháp trên bộ dữ liệu này.
- **Viết báo cáo chi tiết:** Mỗi thành viên sẽ viết một báo cáo chi tiết về thuật toán mình thực hiện, giải thích cặn kẽ từ lý thuyết đến thực hành.

## Mô tả dữ liệu
Dự án sử dụng bộ dữ liệu `[Weather Type Classification]`.
- **Nguồn:** `[https://www.kaggle.com/datasets/nikhil7280/weather-type-classification]`

## Cấu trúc thư mục
Dự án được tổ chức theo cấu trúc gợi ý như sau để dễ dàng quản lý và cộng tác:
```
/
├── data/
│   ├── raw/                  # Chứa dữ liệu thô ban đầu
│   │   └── weather_data.csv
│   └── processed/            # Chứa dữ liệu đã qua xử lý
│       └── cleaned_data.csv
├── notebooks/                # Chứa các file Jupyter Notebook để khám phá, thử nghiệm
│   ├── 1_data_exploration.ipynb
│   ├── 2_svm_model.ipynb
│   └── 3_xgboost_model.ipynb
├── src/                      # Chứa code Python hoàn chỉnh
│   ├── preprocess.py         # Module tiền xử lý dữ liệu
│   ├── train_svm.py          # Script huấn luyện mô hình SVM
│   ├── train_xgboost.py      # Script huấn luyện mô hình XGBoost
|   ├── utils.py
│   └── evaluate.py           # Script đánh giá mô hình
├── requirements.txt          # Liệt kê các thư viện cần thiết
└── README.md
```

## Phân công nhiệm vụ

| Nhiệm vụ | Người thực hiện | Mô tả công việc |
|---|---|---|
| **Task 1: SVM Model** | `[Dũng, Chiến]` | - Phân tích các yêu cầu tiền xử lý dữ liệu riêng cho SVM (ví dụ: scaling).<br>- Xây dựng, huấn luyện và tinh chỉnh mô hình SVM.<br>- Viết báo cáo chi tiết cho thuật toán SVM. |
| **Task 2: XGBoost Model** | `[Dương, Nhi]` | - Phân tích các yêu cầu tiền xử lý dữ liệu cho XGBoost (ví dụ: xử lý categorical).<br>- Xây dựng, huấn luyện và tinh chỉnh mô hình XGBoost.<br>- Viết báo cáo chi tiết cho thuật toán XGBoost. |
| **Task chung: Đánh giá** | `Làm chung` | - Thống nhất các độ đo (metrics) để đánh giá.<br>- Viết script `evaluate.py` chung.<br>- Cùng nhau thực hiện so sánh, rút ra kết luận cuối cùng về hiệu suất hai mô hình. |


## Yêu cầu Báo cáo (Report)
Mỗi báo cáo cần được viết **cực kỳ chi tiết, rõ ràng**, với mục tiêu giúp một người mới có thể đọc và hiểu được toàn bộ quá trình. Nội dung bắt buộc bao gồm:

1.  **Giới thiệu về thuật toán:**
    -   Trình bày khái niệm cốt lõi của thuật toán (SVM là gì? XGBoost là gì?).
    -   Giải thích các khái niệm quan trọng (ví dụ: với SVM là support vectors, kernel trick, margin; với XGBoost là gradient boosting, decision tree, regularization).
    -   Nêu ưu và nhược điểm lý thuyết của thuật toán.

2.  **Tiền xử lý dữ liệu (Data Preprocessing):**
    -   Giải thích tại sao thuật toán này cần các bước tiền xử lý dữ liệu cụ thể.
        -   *Ví dụ cho SVM:* "SVM rất nhạy cảm với sự khác biệt về thang đo của các features, do đó việc chuẩn hóa (Scaling) dữ liệu như StandardScaler là bắt buộc để đảm bảo các feature có đóng góp công bằng vào việc xác định siêu phẳng phân tách..."
        -   *Ví dụ cho XGBoost:* "XGBoost có thể xử lý trực tiếp các giá trị thiếu, tuy nhiên để đảm bảo tính nhất quán, chúng ta đã... Thuật toán cũng yêu cầu các biến categorical phải được mã hóa thành số..."
    -   Liệt kê các bước đã thực hiện và giải thích lý do.

3.  **Xây dựng mô hình:**
    -   Trình bày quá trình huấn luyện mô hình.
    -   Giải thích ý nghĩa của các tham số (hyperparameters) quan trọng đã được tinh chỉnh (ví dụ: `C`, `gamma`, `kernel` cho SVM; `n_estimators`, `max_depth`, `learning_rate` cho XGBoost).
    -   Mô tả phương pháp đã dùng để tìm tham số tốt nhất (ví dụ: GridSearch, RandomizedSearch).

4.  **Kết quả và phân tích:**
    -   Trình bày kết quả đánh giá (accuracy, confusion matrix, ...).
    -   Phân tích sâu về kết quả: Mô hình hoạt động tốt ở điểm nào, yếu ở điểm nào? Tại sao?

5.  **Kết luận:**
    -   Tóm tắt lại quá trình và kết quả đạt được cho thuật toán của mình.

## Deadline
**Deadline hoàn thành toàn bộ dự án và báo cáo: `30/10/2025`**

# ⚠️ Chú ý
  - Mỗi Task tạo 1 branch riêng và làm việc trên branch đó, chừng nào làm ổn hết rồi thì mới merge vô main branch
---
