import numpy as np
import cv2
import os

# --- TÊN FILE BẠN MUỐN KIỂM TRA ---
image_name = '73'  # Chỉ cần tên, không cần .jpg
# -----------------------------------

# Đường dẫn (ĐÃ SỬA, CÓ THÊM '..' ĐỂ ĐI LÙI RA)
img_path = os.path.join('..', 'test_images', f"{image_name}.jpg")
npz_path = os.path.join('..', 'test_output', f"{image_name}.npz")
out_path = os.path.join('..', 'test_output', f"{image_name}_boxed_numbered.jpg") # Đổi tên file output

# 1. Đọc file .npz để lấy tọa độ
print(f"Dang doc file: {npz_path}")
data = np.load(npz_path)
boxes = data['boxes']
print(f"Tim thay {len(boxes)} cai hop (boxes).")

# 2. Đọc ảnh gốc
img = cv2.imread(img_path)
if img is None:
    print(f"LOI: Khong tim thay anh goc tai: {img_path}")
else:
    # 3. Vẽ từng cái hộp VÀ ĐÁNH SỐ
    # Dùng enumerate để có cả chỉ số (i) và giá trị (box)
    for i, box in enumerate(boxes):
        # Lấy tọa độ và ép kiểu về số nguyên
        x1, y1, x2, y2 = [int(b) for b in box]
        
        # Lấy số thứ tự (bắt đầu từ 1)
        number_text = str(i + 1)
        
        # --- BẮT ĐẦU VẼ ---
        
        # A. Vẽ hình chữ nhật (màu xanh lá)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # B. Vẽ số (màu đỏ)
        # Đặt số ở ngay trên góc trái của hộp
        cv2.putText(
            img,
            number_text,
            (x1, y1 - 10), # Vị trí (x, y) của chữ
            cv2.FONT_HERSHEY_SIMPLEX, # Font chữ
            0.9, # Cỡ chữ
            (0, 0, 255), # Màu đỏ (BGR)
            2 # Độ dày
        )

    # 4. Lưu ảnh kết quả ra
    cv2.imwrite(out_path, img)
    print(f"DA LUU ANH KET QUA TAI: {out_path}")

    # 5. (Tùy chọn) Hiển thị ảnh
    # cv2.imshow('Ket Qua Kiem Tra', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()