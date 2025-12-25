import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
import os

warnings.filterwarnings('ignore')

# ==========================================
# 1. NGUỒN DỮ LIỆU (LOAD DATA)
# ==========================================
file_name = 'HeartDiseaseTrain-Test.csv'

# Kiểm tra xem file có tồn tại không để báo lỗi rõ ràng
if not os.path.exists(file_name):
    print(f"LỖI: Không tìm thấy file '{file_name}'!")
    print(f"Vui lòng copy file '{file_name}' vào cùng thư mục với file code này.")
    print(f"Thư mục hiện tại là: {os.getcwd()}")
    exit()

df = pd.read_csv(file_name)
print(f"Dữ liệu gốc: {df.shape}")
print(df.head())

# ==========================================
# 2. TIỀN XỬ LÝ (PREPROCESSING)
# ==========================================
print("\n--- Đang xử lý dữ liệu ---")
# 2.1. Rời rạc hóa (Binning) các biến liên tục
# Các biến số cần chia khoảng: age, blood_pressure, cholesterol, heart_rate...
num_cols = ['age', 'resting_blood_pressure', 'cholestoral', 'Max_heart_rate', 'oldpeak']
df_binned = df.copy()

for col in num_cols:
    try:
        # Chia thành 4 khoảng (Quartiles: Q1, Q2, Q3, Q4)
        df_binned[col] = pd.qcut(df_binned[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4']).astype(str)
        df_binned[col] = df_binned[col].apply(lambda x: f"{col}_{x}")
    except ValueError:
        # Nếu dữ liệu ít giá trị unique (không chia đều 4 phần được), dùng cut (chia đều khoảng giá trị)
        df_binned[col] = pd.cut(df_binned[col], bins=3, labels=['Low', 'Medium', 'High']).astype(str)
        df_binned[col] = df_binned[col].apply(lambda x: f"{col}_{x}")

# 2.2. Xử lý biến phân loại (Categorical)
# Thêm tên cột vào trước giá trị để dễ đọc (VD: 'Male' -> 'sex_Male')
cat_cols = [c for c in df.columns if c not in num_cols and c != 'target']
for col in cat_cols:
    df_binned[col] = df_binned[col].astype(str).apply(lambda x: f"{col}_{x}")

# Xử lý cột Target
df_binned['target'] = df_binned['target'].apply(lambda x: f"target_{x}")

# 2.3. Mã hóa One-Hot (Tạo ma trận giao dịch cho Apriori)
X_binary = pd.get_dummies(df_binned, prefix="", prefix_sep="")
print(f"Kích thước ma trận nhị phân sau xử lý: {X_binary.shape}")

# ==========================================
# 3. APRIORI - TÌM LUẬT KẾT HỢP
# ==========================================
print("\n--- Đang chạy Apriori ---")
# Tìm tập phổ biến (Frequent Itemsets) với min_support = 0.2
frequent_itemsets = apriori(X_binary, min_support=0.2, use_colnames=True)

# Sinh luật kết hợp với Lift >= 1
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values('lift', ascending=False)

print(f"Số lượng luật tìm thấy: {len(rules)}")
if not rules.empty:
    print("\nTop 5 luật kết hợp (theo Lift):")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))
    
    # Lưu luật ra file CSV
    rules.to_csv('heart_disease_association_rules.csv', index=False)
    print("-> Đã lưu luật vào file 'heart_disease_association_rules.csv'")

# ==========================================
# 4. PHÂN CỤM (K-MEANS) & ĐÁNH GIÁ
# ==========================================
print("\n--- Đang chạy K-Means Clustering ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_binary)

ks = range(2, 11)
inertias = []
sils = []

for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, labels))

best_k = ks[np.argmax(sils)]
print(f"Số cụm (k) tối ưu dựa trên Silhouette Score: {best_k}")

# Vẽ biểu đồ Elbow và Silhouette
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Số lượng cụm (k)')
ax1.set_ylabel('Inertia (SSE)', color=color)
ax1.plot(ks, inertias, marker='o', color=color, label='Inertia')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Silhouette Score', color=color)  
ax2.plot(ks, sils, marker='s', color=color, label='Silhouette')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Đánh giá K-Means: Elbow Method và Silhouette Score')
fig.tight_layout()
plt.savefig('clustering_evaluation.png')
print("-> Đã lưu biểu đồ vào file 'clustering_evaluation.png'")
plt.show() 
# ==========================================
# 5. KẾT QUẢ CUỐI CÙNG & SO SÁNH
# ==========================================
# Chạy lại K-Means với k tối ưu
final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = final_km.fit_predict(X_scaled)

print("\n--- Thống kê kết quả phân cụm ---")
print(df['Cluster'].value_counts())

# So sánh Cluster với nhãn Target gốc (Quan trọng để báo cáo)
print("\n--- Bảng so sánh giữa Cluster (Máy chia) và Bệnh thực tế (Target) ---")
crosstab = pd.crosstab(df['Cluster'], df['target'])
print(crosstab)

df.to_csv('HeartDisease_Clustered.csv', index=False)
print("\n-> Đã lưu dữ liệu phân cụm vào file 'HeartDisease_Clustered.csv'")
print("HOÀN THÀNH!")