import pandas as pd
import numpy as np

def clean_bank_dataset(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()

    # 1. Chuẩn hóa các đại diện của giá trị thiếu (Missing values)
    # Chuyển tất cả về dạng np.nan để dễ dàng quản lý
    df = df.replace(["none", "None", "N/A", "unknown", ""], np.nan)

    # 2. Xử lý các cột văn bản (Categorical columns)
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                        'loan', 'contact', 'month', 'poutcome', 'deposit']

    for col in categorical_cols:
        if col in df.columns:
            # Xóa khoảng trắng thừa và chuẩn hóa viết hoa chữ cái đầu (Title Case)
            df[col] = df[col].astype(str).str.strip().str.title()
            
            # Xử lý riêng các giá trị Missing sau khi chuẩn hóa
            # Riêng poutcome có thể đổi Nan thành "New Customer"
            if col == 'poutcome':
                df[col] = df[col].replace('Nan', 'New Customer')
            else:
                df[col] = df[col].replace('Nan', 'Unknown')

    # 3. Ép kiểu dữ liệu về số thực cho các cột định lượng
    float_cols = ['age', 'balance', 'duration']
    for col in float_cols:
        df[col] = df[col].astype('float64')

    numeric_cols = ['day', 'campaign', 'pdays', 'previous']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # 4. Tối ưu bộ nhớ bằng cách chuyển sang kiểu 'category'
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    return df

# Cách sử dụng
df_raw = pd.read_csv('bank_data.csv')
df_cleaned = clean_bank_dataset(df_raw)

print(df_cleaned.head(10))