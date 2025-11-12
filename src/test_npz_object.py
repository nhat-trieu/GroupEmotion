import numpy as np
import os

# --- SỬA DÒNG NÀY ---
file_path = r"features_objects_FULL_zip\train\Positive\3300.npz"

# --- KHÔNG SỬA GÌ Ở DƯỚI ---

if not os.path.exists(file_path):
    print(f"ERROR: File not found at: {file_path}")
    print("Check your folder name 'features_objects_FULL_zip' again.")
else:
    print(f"Reading file: {file_path}")
    data = np.load(file_path)
    
    # Print all keys inside the .npz file
    print(f"Keys in file: {list(data.keys())}")
    
    # Unpack the data
    objects_features = data['objects']
    boxes = data['boxes']
    classes_ids = data['classes']
    
    print("\n--- TEST RESULT (OBJECTS) ---")
    print(f"Objects found: {len(objects_features)}")
    print(f"Features shape (K, 2048): {objects_features.shape}") 
    print(f"Boxes shape (K, 4): {boxes.shape}")
    print(f"Class IDs: {classes_ids}")

    if len(objects_features) > 0:
        print("\nSample feature (first 5 values):")
        print(objects_features[0][:5])
    else:
        print("This file contains no objects.")