import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
import shutil
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """Load dataset dari file CSV"""
    print("=" * 60)
    print("STEP 1: Loading Dataset")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"✓ Dataset loaded successfully!")
    print(f"  Shape: {df.shape}")
    print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    return df


def handle_missing_values(df):
    """Handle missing values dengan median"""
    print("\n" + "=" * 60)
    print("STEP 2: Handle Missing Values")
    print("=" * 60)
    
    missing_before = df.isnull().sum().sum()
    print(f"Total missing values before: {missing_before}")
    
    # Fill missing values dengan median untuk setiap kolom
    for col in df.columns:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"{col} filled with median: {median_val:.2f}")
    
    missing_after = df.isnull().sum().sum()
    print(f"Total missing values after: {missing_after}")
    
    return df


def drop_duplicates(df):
    """Drop duplicate rows"""
    print("\n" + "=" * 60)
    print("STEP 3: Drop Duplicates")
    print("=" * 60)
    
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    dropped = before - after
    
    print(f"Rows before: {before}")
    print(f"Rows after: {after}")
    print(f"Duplicates dropped: {dropped}")
    
    return df


def scale_features(df):
    """Scale ALL features except target (Outcome)"""
    print("\n" + "=" * 60)
    print("STEP 4: Scale Numeric Features")
    print("=" * 60)
    
    # Features to scale: semua kecuali Outcome
    features_to_scale = [col for col in df.columns if col != 'Outcome']
    
    print("Features to scale:")
    for feat in features_to_scale:
        print(f"  - {feat}")
    
    if features_to_scale:
        scaler = StandardScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
        print(f"\n{len(features_to_scale)} features scaled using StandardScaler")
    
    return df


def split_data(df, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    print("\n" + "=" * 60)
    print("STEP 5: Train-Test Split")
    print("=" * 60)
    
    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nData split completed:")
    print(f"  Train size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"\n  Train target distribution:")
    print(f"    Class 0 (No Diabetes): {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
    print(f"    Class 1 (Diabetes): {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test, df


def apply_smote(X_train, y_train):
    """Apply SMOTE untuk handle class imbalance"""
    print("\n" + "=" * 60)
    print("STEP 6: Apply SMOTE (Oversampling)")
    print("=" * 60)
    
    print(f"Before SMOTE:")
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"  Class 0 (No Diabetes): {counts[0]} ({counts[0]/len(y_train)*100:.1f}%)")
    print(f"  Class 1 (Diabetes): {counts[1]} ({counts[1]/len(y_train)*100:.1f}%)")
    print(f"  Total samples: {len(y_train)}")
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\nAfter SMOTE:")
    unique, counts = np.unique(y_train_resampled, return_counts=True)
    print(f"  Class 0 (No Diabetes): {counts[0]} ({counts[0]/len(y_train_resampled)*100:.1f}%)")
    print(f"  Class 1 (Diabetes): {counts[1]} ({counts[1]/len(y_train_resampled)*100:.1f}%)")
    print(f"  Total samples: {len(y_train_resampled)}")
    print(f"\nSMOTE applied successfully!")
    print(f"  Samples added: {len(y_train_resampled) - len(y_train)}")
    
    return X_train_resampled, y_train_resampled


def save_data(X_train, X_test, y_train, y_test, df_processed, output_dir):
    """Save preprocessed data"""
    print("\n" + "=" * 60)
    print("STEP 7: Save Preprocessed Data")
    print("=" * 60)
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save files
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False, header=['Outcome'])
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False, header=['Outcome'])
    df_processed.to_csv(os.path.join(output_dir, 'diabetes_preprocessed.csv'), index=False)
    
    print(f"All files saved to: {output_dir}")
    print(f"\n  Files saved:")
    print(f"    - X_train.csv: {X_train.shape}")
    print(f"    - X_test.csv: {X_test.shape}")
    print(f"    - y_train.csv: {y_train.shape}")
    print(f"    - y_test.csv: {y_test.shape}")
    print(f"    - diabetes_preprocessed.csv: {df_processed.shape}")


def copy_to_modelling_folder(source_dir, dest_dir):
    """FUNGSI BARU: Copy files ke folder Membangun_model"""
    print("\n" + "=" * 60)
    print("STEP 8: Copy Files to Membangun_model Folder")
    print("=" * 60)
    
    # Create destination directory if not exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # List of files to copy
    files_to_copy = [
        'X_train.csv',
        'X_test.csv',
        'y_train.csv',
        'y_test.csv',
        'diabetes_preprocessed.csv'
    ]
    
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print("\nCopying files:")
    
    for filename in files_to_copy:
        source_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(dest_dir, filename)
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, dest_file)
            print(f"  ✓ {filename} copied successfully")
        else:
            print(f"  ✗ {filename} not found in source")
    
    print(f"\n✓ All files copied to: {dest_dir}")


def main():
    """Main preprocessing pipeline"""
    print("\n" + "=" * 60)
    print("AUTOMATED PREPROCESSING - DIABETES DATASET")
    print("=" * 60)
    print("Author: Angga Yulian Adi Pradana")
    print("Class: Membangun Sistem Machine Learning")
    print("=" * 60)

    # Deteksi apakah script dijalankan di GitHub Actions atau lokal
    is_github_actions = os.getenv('GITHUB_ACTIONS') == 'true'
    
    if is_github_actions:
        # Path untuk GitHub Actions
        input_path = '../diabetes_raw/diabetes.csv'
        output_dir = './diabetes_preprocessing/'
        print("\n🔧 Running in GitHub Actions mode")
    else:
        # Path untuk lokal
        input_path = r"D:\Perkuliahan\Asah led by Dicoding\Submission Proyek\SMSML_Angga\Eksperimen_SML_Angga-Yulian-Adi-Pradana\diabetes_raw\diabetes.csv"
        output_dir = './diabetes_preprocessing/'
        modelling_output_dir = r'D:\Perkuliahan\Asah led by Dicoding\Submission Proyek\SMSML_Angga\Membangun_model\diabetes_preprocessing'
        print("\n🔧 Running in Local mode")
    
    # Pipeline
    df = load_data(input_path)
    df = handle_missing_values(df)
    df = drop_duplicates(df)
    df = scale_features(df)
    X_train, X_test, y_train, y_test, df_processed = split_data(df)
    
    # APPLY SMOTE untuk handle class imbalance
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # Save data
    save_data(X_train_resampled, X_test, y_train_resampled, y_test, df_processed, output_dir)
    
    # COPY FILES KE FOLDER MEMBANGUN_MODEL (hanya di lokal)
    if not is_github_actions:
        copy_to_modelling_folder(output_dir, modelling_output_dir)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nSUMMARY:")
    print("✓ Missing values handled with median")
    print("✓ Duplicates removed")
    print("✓ All features scaled (StandardScaler)")
    print("✓ Train-test split (80-20)")
    print("✓ Class imbalance handled with SMOTE")
    print("✓ Data saved to preprocessing folder")
    if not is_github_actions:
        print("✓ Data copied to Membangun_model folder")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
