import os
import shutil

def delete_pycache_dirs(base_path="."):
    count = 0
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                full_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(full_path)
                    print(f"✅ Deleted: {full_path}")
                    count += 1
                except PermissionError:
                    print(f"⚠️ Skipped (no permission): {full_path}")
                except Exception as e:
                    print(f"❌ Error deleting {full_path}: {e}")
    if count == 0:
        print("No __pycache__ directories were deleted.")
    else:
        print(f"✅ Done! Deleted {count} __pycache__ directories.")

if __name__ == "__main__":
    delete_pycache_dirs()
