Dưới đây là phiên bản được viết lại, chuyên nghiệp hơn, trình bày rõ ràng và bổ sung thêm các hướng dẫn cần thiết để chạy dự án.

---

# 🌦️ Phân Loại Thời Tiết: SVM vs. XGBoost

> **Weather Type Classification on Tabular Dataset**

Dự án này tập trung vào việc xây dựng, tối ưu hóa và so sánh hiệu suất của hai thuật toán học máy phổ biến là **Logisitic Regression**, **Support Vector Machine (SVM)** và **XGBoost** (Extreme Gradient Boosting) trong bài toán phân loại thời tiết dựa trên dữ liệu dạng bảng.

---

## 📑 Mục Lục
1. [Giới thiệu dự án](#-giới-thiệu-dự-án)
2. [Dữ liệu](#-dữ-liệu)
3. [Cấu trúc dự án](#-cấu-trúc-dự-án)
4. [Cài đặt và Sử dụng](#-cài-đặt-và-sử-dụng)
5. [Phân công nhiệm vụ](#-phân-công-nhiệm-vụ)
6. [Yêu cầu báo cáo](#-yêu-cầu-báo-cáo)
7. [Lịch trình](#-lịch-trình)

---

## 🚀 Giới thiệu dự án

Mục tiêu chính của dự án là giải quyết bài toán phân loại đa lớp (Multi-class classification) để dự đoán các loại hình thời tiết. Quy trình thực hiện bao gồm:

1.  **Tiền xử lý dữ liệu (Data Preprocessing):** Làm sạch dữ liệu, xử lý missing values, mã hóa (Encoding) và chuẩn hóa (Scaling).
2.  **Mô hình hóa (Modeling):**
    *   **Logistic Regression:** Mô hình tuyến tính sử dụng hàm Softmax để phân loại đa lớp, yêu cầu chuẩn hóa dữ liệu và thường được dùng làm baseline do tính đơn giản và khả năng diễn giải tốt.
    *   **SVM:** Tập trung vào việc xây dựng siêu phẳng phân tách tối ưu, yêu cầu kỹ lưỡng về scaling dữ liệu.
    *   **XGBoost:** Sử dụng kỹ thuật boosting trên cây quyết định, tập trung vào tốc độ và hiệu suất cao.
3.  **Tối ưu tham số (Hyperparameter Tuning):** Sử dụng GridSearch hoặc RandomizedSearch để tìm bộ tham số tốt nhất.
4.  **Đánh giá & So sánh:** Phân tích kết quả dựa trên Accuracy, F1-Score, Precision, Recall và Confusion Matrix.

---

## 📊 Dữ liệu

Dự án sử dụng bộ dữ liệu **Weather Type Classification** từ Kaggle.
*   **Nguồn dữ liệu:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/nikhil7280/weather-type-classification)
*   **Loại dữ liệu:** Dạng bảng (Tabular data).
*   **Target:** Các loại hình thời tiết (Ví dụ: Sunny, Rainy, Cloudy, Snowy...).

---

## 📂 Cấu trúc dự án

```bash
Weather-Type-Prediction/
├── data/
│   └── raw/                    # Chứa dữ liệu thô tải về từ Kaggle (weather_data.csv)
├── notebooks/                  # Jupyter Notebooks cho từng giai đoạn
│   ├── 1_data_exploration.ipynb    # Khám phá dữ liệu sơ bộ
│   ├── 01_EDA_Preprocessing.ipynb  # Phân tích sâu và tiền xử lý
│   └── 02_ModelingSelection.ipynb  # Huấn luyện, tinh chỉnh và so sánh mô hình
├── src/                        # Mã nguồn Python (để tái sử dụng)
│   ├── preprocess.py           # Các hàm xử lý, làm sạch dữ liệu
│   └── utils.py                # Các hàm hỗ trợ (đánh giá, vẽ biểu đồ...)
├── requirements.txt            # Danh sách thư viện cần thiết
└── README.md                   # Tài liệu dự án
```

---

## 🛠 Cài đặt và Sử dụng

1.  **Clone dự án:**
    ```bash
    git clone https://github.com/danthuong/Weather-type-prediction-on-tabular-dataset.git
    cd Weather-type-prediction-on-tabular-dataset
    ```

2.  **Cài đặt môi trường:**
    Khuyên dùng môi trường ảo (Virtual Environment):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Chạy Notebook:**
    Mở các file trong thư mục `notebooks/` theo thứ tự đã đánh số để theo dõi quy trình.

---

## 👥 Phân công nhiệm vụ

| Task | Thành viên | Mô tả công việc chi tiết |
| :--- | :--- | :--- |
| **Task 1: SVM Model** | `[Dũng, Chiến]` | - Phân tích đặc thù dữ liệu cho SVM + LogisticRegression.<br>- Huấn luyện và Tuning SVM.<br>- Viết báo cáo chuyên sâu về SVM. |
| **Task 2: XGBoost Model** | `[Dương, Nhi]` | - Phân tích đặc thù dữ liệu cho XGBoost.<br>- Huấn luyện và Tuning XGBoost.<br>- Viết báo cáo chuyên sâu về XGBoost. |
| **Task 3: Đánh giá chung** | `Toàn team` | - Thống nhất metrics đánh giá.<br>- Viết script so sánh.<br>- Tổng hợp kết quả và viết kết luận chung. |

---

## 📝 Yêu cầu báo cáo

Báo cáo cần được trình bày **chi tiết, mang tính học thuật và giải thích rõ ràng** để người đọc (kể cả người mới) có thể hiểu được. Cấu trúc bắt buộc:

### 1. Giới thiệu thuật toán
*   **Khái niệm cốt lõi:** Định nghĩa SVM/XGBoost là gì?
*   **Cơ chế hoạt động:**
    *   *Logistic Regression:* Linear decision boundary, Sigmoid/Softmax, Cross-Entropy loss, L2 regularization.
    *   *SVM:* Support vectors, Margin, Kernel Trick ($C$, $\gamma$...).
    *   *XGBoost:* Gradient Boosting, Decision Trees, Regularization, Loss function.
*   **Ưu/Nhược điểm lý thuyết:** Khi nào nên dùng?

### 2. Tiền xử lý dữ liệu (Data Preprocessing)
*   **Lý do thực hiện:** Tại sao thuật toán này lại cần bước xử lý đó?
    *   *Ví dụ SVM:* Tại sao phải dùng StandardScaler/MinMaxScaler?
    *   *Ví dụ XGBoost:* Xử lý biến category (One-Hot vs Label Encoding) ảnh hưởng thế nào?
*   **Quy trình:** Liệt kê các bước làm sạch và biến đổi dữ liệu đã áp dụng.

### 3. Xây dựng mô hình (Modeling)
*   **Quá trình huấn luyện:** Các bước train model.
*   **Hyperparameters:** Giải thích ý nghĩa các tham số quan trọng đã tinh chỉnh.
*   **Phương pháp Tuning:** GridSearch hay RandomizedSearch? Tại sao chọn không gian tham số đó?

### 4. Kết quả & Phân tích (Evaluation)
*   **Kết quả định lượng:** Bảng số liệu (Accuracy, F1-Score...).
*   **Kết quả định tính:** Confusion Matrix, ROC Curve.
*   **Phân tích sâu:** Model nhận diện tốt lớp nào? Kém lớp nào? Tại sao (do dữ liệu mất cân bằng hay do đặc trưng)?

### 5. Kết luận
*   Tổng kết lại hiệu quả của thuật toán đối với bộ dữ liệu này.

---

## 📅 Deadline
*   **Deadline hoàn thành:** `23/11/2025`