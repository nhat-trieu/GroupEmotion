import numpy as np
import os


# (Ví dụ,  lấy file 3300.npz trong thư mục của bạn)
file_path = r"..\features_face_FULL_zip\train\Positive\3300.npz"

# Kiểm tra xem file có tồn tại không
if not os.path.exists(file_path):
    print(f"LOI: Khong tim thay file tai: {file_path}")
    print("Dam bao ban dat file script nay o thu muc 'GroupEmotion' (ngang hang voi 'features_face_FULL_zip')")
else:
    print(f"Dang doc file: {file_path}")
    
    # Doc va hien thi
    data = np.load(file_path)
    faces = data['faces']
    boxes = data['boxes']

    print("\n--- KET QUA ---")
    print(f"So khuon mat: {len(faces)}")
    print(f"Shape faces: {faces.shape}")  # Mong đợi (N, 512)
    print(f"Shape boxes: {boxes.shape}")  # Mong đợi (N, 4)
    print(f"\nSample embedding (5 gia tri dau):")
    
    if len(faces) > 0:
        print(faces[0][:5])
    else:
        print("File nay khong co khuon mat nao.")