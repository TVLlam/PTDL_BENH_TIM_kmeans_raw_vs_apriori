# BÁO CÁO BÀI TẬP: KHAI PHÁ DỮ LIỆU TIM MẠCH (HEART DISEASE)

Dự án này thực hiện quy trình Khai phá dữ liệu (Data Mining) trên bộ dữ liệu `HeartDiseaseTrain-Test.csv` nhằm tìm kiếm các quy luật kết hợp giữa các triệu chứng và phân nhóm bệnh nhân.

## 1. Mục tiêu
- **Tìm luật kết hợp (Association Rules):** Sử dụng thuật toán **Apriori** để tìm mối liên hệ giữa các chỉ số (tuổi, huyết áp, đau ngực...) và khả năng mắc bệnh.
- **Phân cụm dữ liệu (Clustering):** Sử dụng thuật toán **K-Means** để nhóm các bệnh nhân có đặc điểm giống nhau và so sánh với nhãn bệnh thực tế.

## 2. Công nghệ & Thư viện sử dụng
- Ngôn ngữ: Python
- Thư viện chính:
  - `pandas`, `numpy`: Xử lý dữ liệu.
  - `mlxtend`: Thuật toán Apriori (Khai phá luật kết hợp).
  - `scikit-learn`: Thuật toán K-Means, chuẩn hóa dữ liệu và đánh giá (Silhouette Score).
  - `matplotlib`: Vẽ biểu đồ đánh giá.

## 3. Quy trình thực hiện (Methodology)

### Bước 1: Tiền xử lý dữ liệu (Preprocessing)
Do thuật toán Apriori yêu cầu dữ liệu dạng giao dịch (Category), quy trình xử lý bao gồm:
- **Rời rạc hóa (Binning):** Chuyển các biến liên tục (`age`, `cholesterol`, `blood_pressure`...) thành các khoảng giá trị (Ví dụ: `age_Q1`, `age_Q2`...).
- **Mã hóa One-Hot:** Chuyển đổi toàn bộ dữ liệu sang dạng nhị phân (0/1) để tạo ma trận giao dịch.

### Bước 2: Khai phá luật kết hợp (Apriori)
- Thiết lập độ hỗ trợ tối thiểu (`min_support`) = 0.2.
- Sinh luật với độ nâng (`Lift`) >= 1.0.
- **Kết quả:** Tìm ra các luật mạnh chỉ ra mối quan hệ giữa triệu chứng và bệnh lý (Lưu trong file `heart_disease_association_rules.csv`).

### Bước 3: Phân cụm (K-Means)
- Sử dụng phương pháp **Elbow** và **Silhouette Score** để xác định số cụm tối ưu ($k$).
- **Kết quả:** Số cụm tối ưu là **$k=2$**, tương ứng với nhóm "Có bệnh" và "Không bệnh".

## 4. Hướng dẫn cài đặt và Chạy

1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install pandas numpy matplotlib scikit-learn mlxtend
Đảm bảo file dữ liệu HeartDiseaseTrain-Test.csv nằm cùng thư mục với code.

2.Chạy chương trình:
 ```bash
   ppython final_solution.py

## 5. Kết quả đầu ra
Sau khi chạy, chương trình sẽ sinh ra các file:

heart_disease_association_rules.csv: Bảng chứa các luật kết hợp tìm được.

HeartDisease_Clustered.csv: Dữ liệu gốc kèm theo nhãn phân cụm (Cluster).

clustering_evaluation.png: Biểu đồ đánh giá Elbow và Silhouette.
