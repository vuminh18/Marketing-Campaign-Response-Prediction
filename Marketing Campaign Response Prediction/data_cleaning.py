import pandas as pd
import numpy as np

def handle_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

def clean_bank_dataset(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()

    # --- BƯỚC 1: Chuẩn hóa giá trị thiếu & Logic nghiệp vụ ---
    # (Giữ nguyên ý tưởng của bạn nhưng làm chuẩn hơn)
    df = df.replace(["none", "None", "N/A", "unknown", ""], np.nan)
    
    # Logic quan trọng: Chỉ pdays == -1 mới chắc chắn là New Customer
    if 'poutcome' in df.columns and 'pdays' in df.columns:
        df.loc[(df['poutcome'].isna()) & (df['pdays'] == -1), 'poutcome'] = 'New Customer'

    # --- BƯỚC 2: Xử lý cột văn bản & Ép kiểu Category ---
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing',
                        'loan', 'contact', 'month', 'poutcome', 'deposit']

    for col in categorical_cols:
        if col in df.columns:
            # Chuyển Target 'deposit' về số 0/1 trước khi Title() để tránh lỗi
            if col == 'deposit':
                df[col] = df[col].astype(str).str.lower().map({'yes': 1, 'no': 0})
            else:
                df[col] = df[col].astype(str).str.strip().str.title()
                df[col] = df[col].replace('Nan', 'Unknown')
            
            df[col] = df[col].astype('category')

    # --- BƯỚC 3: Xử lý ngoại lai (Outliers) - ĐÂY LÀ PHẦN BẠN ĐANG THIẾU ---
    # Phải gọt dũa dữ liệu số trước khi ép kiểu cuối cùng
    cols_to_fix = ['balance', 'duration', 'campaign']
    for col in cols_to_fix:
        if col in df.columns:
            df = handle_outliers_iqr(df, col)

    # --- BƯỚC 4: Ép kiểu số chuẩn ---
    float_cols = ['age', 'balance', 'duration']
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

    numeric_cols = ['day', 'campaign', 'pdays', 'previous']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    return df
# Ví dụ sử dụng
df_raw = pd.read_csv('bank_data.csv')
df_cleaned = clean_bank_dataset(df_raw)
