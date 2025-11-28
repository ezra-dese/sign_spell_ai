import os

def clear_dataset():
    """
    Clears the dataset and model files to start fresh.
    """
    data_dir = "../data"
    dataset_path = os.path.join(data_dir, "dataset.csv")
    model_path = os.path.join(data_dir, "model.pkl")
    
    files_deleted = []
    
    # Delete dataset.csv
    if os.path.exists(dataset_path):
        os.remove(dataset_path)
        files_deleted.append("dataset.csv")
        print(f"✓ Deleted {dataset_path}")
    else:
        print(f"⚠ {dataset_path} not found (already clean)")
    
    # Delete model.pkl
    if os.path.exists(model_path):
        os.remove(model_path)
        files_deleted.append("model.pkl")
        print(f"✓ Deleted {model_path}")
    else:
        print(f"⚠ {model_path} not found (already clean)")
    
    if files_deleted:
        print(f"\n✅ Cleared {len(files_deleted)} file(s). You can now run data_collector.py to collect fresh data.")
    else:
        print("\n✅ Nothing to clear. Ready to collect data.")

if __name__ == "__main__":
    print("=" * 50)
    print("DATASET CLEANER")
    print("=" * 50)
    print("This will delete:")
    print("  - data/dataset.csv (training data)")
    print("  - data/model.pkl (trained model)")
    print()
    
    confirm = input("Are you sure? (yes/no): ").strip().lower()
    
    if confirm in ['yes', 'y']:
        print()
        clear_dataset()
    else:
        print("\n❌ Cancelled. No files were deleted.")
