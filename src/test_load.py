
import numpy as np

# Doc file .npz
data = np.load('D:/KLTN\GroupEmotion/test_output/73.npz')
faces = data['faces']
boxes = data['boxes']

print(f"So khuon mat: {len(faces)}")
print(f"Shape faces: {faces.shape}")  # Should be (N, 512)
print(f"Shape boxes: {boxes.shape}")  # Should be (N, 4)
print(f"Sample embedding: {faces[0][:5]}...")  # First 5 values