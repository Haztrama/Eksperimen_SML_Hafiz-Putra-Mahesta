import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def preprocess_data():
    # --- 1. SETUP PATH ---
    # Mendapatkan lokasi file script ini berada
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Path ke data mentah (naik satu level ke dataset_raw)
    raw_data_path = os.path.join(base_dir, '..', 'bank_churn_raw', 'bank_churn.csv')
    # Path output (Folder khusus sesuai rubrik)
    output_dir = os.path.join(base_dir, 'bank_churn_preprocessing') 

    print("Memulai Otomatisasi Preprocessing Bank Churn...")
    
    # --- 2. LOAD DATA ---
    if not os.path.exists(raw_data_path):
        print(f"Error: Dataset tidak ditemukan di {raw_data_path}")
        return

    df = pd.read_csv(raw_data_path)
    
    # --- 3. DATA CLEANING & ENCODING ---
    # Hapus kolom ID
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    # Encoding (Gender & Geography)
    le = LabelEncoder()
    # Tips: Loop untuk mendeteksi kolom object otomatis
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = le.fit_transform(df[col])
        print(f"Kolom '{col}' telah di-encode.")
    
    # --- 4. SCALING & SPLITTING ---
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Gabung kembali
    df_clean = pd.concat([X_scaled, y], axis=1)
    
    # Split 80-20
    train, test = train_test_split(df_clean, test_size=0.2, random_state=42)
    
    # --- 5. SIMPAN DATA ---
    os.makedirs(output_dir, exist_ok=True)
    
    train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print("\nSUKSES!")
    print(f"Data Train tersimpan di: {os.path.join(output_dir, 'train.csv')}")
    print(f"Data Test tersimpan di: {os.path.join(output_dir, 'test.csv')}")

if __name__ == "__main__":
    preprocess_data()