import os
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import pickle
import logging
from pathlib import Path

# --- CẤU HÌNH NHÃN VÀ KÍCH THƯỚC ---
# (Tự động map tên thư mục sang số)
LABEL_MAP = {
    'Negative': 0,
    'Neutral': 1,
    'Positive': 2
}
# Kích thước (feature dim) của từng loại
FEATURE_DIMS = {
    'face': 512,
    'human': 2048,
    'object': 2048,
    'scene': 2048  # ResNet50
}
# Tên file .npz key
FEATURE_KEYS = {
    'face': 'faces',
    'human': 'humans',
    'object': 'objects',
    'scene': 'scene'
}

# --- THỨ TỰ NODE TRONG GRAPH 

NODE_TYPES = {
    'scene': 0,
    'object': 1,
    'human': 2,
    'face': 3
}
# ---------------------------------------------

def setup_logging():
    """Thiết lập logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('build_graph.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("Bat dau xay dung graph...")

def load_feature(base_path, feature_type, split, label_name, stem):
    """Tải 1 file .npz cụ thể"""
    try:
        # Tên thư mục  (ví dụ: 'features_face_FULL_zip')
        folder_name = f'features_{feature_type}_FULL_zip'
        key = FEATURE_KEYS[feature_type]
        
        file_path = base_path / folder_name / split / label_name / f"{stem}.npz"
        
        if not file_path.exists():
            # Trả về mảng rỗng với shape chuẩn
            return np.empty((0, FEATURE_DIMS[feature_type]), dtype=np.float32)
            
        data = np.load(file_path)
        
        if key not in data:
            logging.warning(f"Key '{key}' not in {file_path}")
            return np.empty((0, FEATURE_DIMS[feature_type]), dtype=np.float32)

        return data[key]

    except Exception as e:
        logging.error(f"Loi khi tai {feature_type} cua file {stem}: {e}")
        return np.empty((0, FEATURE_DIMS[feature_type]), dtype=np.float32)

def build_single_graph(paths, stem, label_id):
    
    # 1. Tải 4 "nguyên liệu" (face, human, object, scene)
    face_features = load_feature(paths['base'], 'face', paths['split'], paths['label'], stem)
    human_features = load_feature(paths['base'], 'human', paths['split'], paths['label'], stem)
    object_features = load_feature(paths['base'], 'object', paths['split'], paths['label'], stem)
    scene_feature = load_feature(paths['base'], 'scene', paths['split'], paths['label'], stem)

    # Đếm số lượng node của từng loại
    n_faces = face_features.shape[0]
    n_humans = human_features.shape[0]
    n_objects = object_features.shape[0]
    
    # Scene luôn là 1 node, ngay cả khi file .npz bị lỗi (sẽ là mảng (0, 2048))
    # Chúng ta sẽ thay thế nó bằng vector 0 nếu nó rỗng
    if scene_feature.shape[0] == 0:
        logging.warning(f"File {stem} thieu 'scene', dung vector 0 thay the.")
        scene_feature = np.zeros((1, FEATURE_DIMS['scene']), dtype=np.float32)
    n_scene = 1 # Luôn có 1 node scene
    
    total_nodes = n_scene + n_objects + n_humans + n_faces
    
    # Nếu không có node nào (trừ scene), graph rỗng, bỏ qua
    if total_nodes <= 1: 
        logging.warning(f"File {stem} khong co node nao, bo qua.")
        return None

    # 2. Tạo "Bàn tiệc" (Node Features `x`)
    # `x` là một ma trận (total_nodes, D)
    # Vì D của 4 món khác nhau (512, 2048...), chúng ta sẽ:
    # a. Tạo một ma trận `x` (total_nodes, D_max) với D_max = 2048
    # b. "Nhồi" các features vào, phần còn thiếu đệm (pad) bằng số 0
    
    x = torch.zeros((total_nodes, FEATURE_DIMS['human']), dtype=torch.float32) # D_max = 2048
    node_type_ids = torch.full((total_nodes,), -1, dtype=torch.long) # Để lưu loại node
    
    current_idx = 0
    
    #   (Scene) - 1 node
    x[current_idx] = torch.from_numpy(scene_feature[0])
    node_type_ids[current_idx] = NODE_TYPES['scene']
    current_idx += n_scene
    
    #  (Objects) - n_objects nodes
    if n_objects > 0:
        x[current_idx : current_idx + n_objects] = torch.from_numpy(object_features)
        node_type_ids[current_idx : current_idx + n_objects] = NODE_TYPES['object']
        current_idx += n_objects
    
    #   (Humans) - n_humans nodes
    if n_humans > 0:
        x[current_idx : current_idx + n_humans] = torch.from_numpy(human_features)
        node_type_ids[current_idx : current_idx + n_humans] = NODE_TYPES['human']
        current_idx += n_humans
        
    #   (Faces) - n_faces nodes
    if n_faces > 0:
        #  (512-chiều) mỏng hơn (2048-chiều), nên ta đệm 0
        face_tensor = torch.from_numpy(face_features)
        x[current_idx : current_idx + n_faces, :FEATURE_DIMS['face']] = face_tensor
        node_type_ids[current_idx : current_idx + n_faces] = NODE_TYPES['face']
        current_idx += n_faces
        
    # 3. Tạo "Quan hệ" (Edge Index)
    # Paper GNN Multi-Cue dùng "Complete Graph" (đồ thị đủ)
    # Tức là: TẤT CẢ node đều kết nối với TẤT CẢ các node khác.
    
    # Tạo 2 danh sách `source_nodes` và `target_nodes`
    src = []
    dst = []
    for i in range(total_nodes):
        for j in range(total_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
                
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # 4. Tạo Nhãn (Label `y`)
    y = torch.tensor(label_id, dtype=torch.long)
    
    # 5. Đóng gói 
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    graph_data.node_type = node_type_ids # Lưu thêm loại node để GNN biết
    graph_data.stem = stem # Lưu tên file
    
    return graph_data


def main():
    setup_logging()
    
    # Đường dẫn này ('..') nghĩa là "đi lùi 1 bước" từ `src` ra `GroupEmotion`
    base_path = Path('..')
    output_dir = base_path / 'graphs'
    output_dir.mkdir(exist_ok=True)
    
    all_graphs = {'train': [], 'val': [], 'test': []}
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        logging.info(f"--- Bat dau xu ly thu muc: {split} ---")
        
        split_path = base_path / 'features_face_FULL_zip' / split
        if not split_path.exists():
            logging.warning(f"Khong tim thay thu muc {split_path}, bo qua.")
            continue
            
        for label_name in LABEL_MAP.keys(): # 'Positive', 'Negative', 'Neutral'
            label_id = LABEL_MAP[label_name]
            label_path = split_path / label_name
            
            if not label_path.exists():
                logging.warning(f"Khong tim thay thu muc {label_path}, bo qua.")
                continue
                
            logging.info(f"Dang quet: {split}/{label_name}...")
            
            # Lấy tên của TẤT CẢ file .npz trong thư mục "face"
            # (Chúng ta lấy "face" làm chuẩn)
            image_stems = [f.stem for f in label_path.glob('*.npz')]
            
            if not image_stems:
                logging.warning(f"Khong tim thay file .npz nao trong {label_path}")
                continue

            # Bắt đầu (graph)
            for stem in tqdm(image_stems, desc=f"Xay dung {split}/{label_name}"):
                
                paths = {'base': base_path, 'split': split, 'label': label_name}
                graph = build_single_graph(paths, stem, label_id)
                
                if graph is not None:
                    all_graphs[split].append(graph)
    
    logging.info("--- TOM TAT KET QUA ---")
    logging.info(f"Tong so graph 'train': {len(all_graphs['train'])}")
    logging.info(f"Tong so graph 'val': {len(all_graphs['val'])}")
    logging.info(f"Tong so graph 'test': {len(all_graphs['test'])}")
    
    # Lưu  vào 1 file duy nhất
    output_file = output_dir / 'graph_data_v1.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_graphs, f)
        
    logging.info(f"--- HOAN TAT! ---")
    logging.info(f"Tat ca graph da duoc luu tai: {output_file}")


if __name__ == '__main__':
    main()