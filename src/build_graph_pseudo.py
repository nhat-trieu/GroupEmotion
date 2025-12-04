import os
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import pickle
import logging
from pathlib import Path
import argparse

# --- CẤU HÌNH ---
LABEL_MAP = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
# Kích thước đặc trưng
FEATURE_DIMS = {'face': 512, 'human': 2048, 'object': 2048, 'scene': 2048}
FEATURE_KEYS = {'face': 'faces', 'human': 'humans', 'object': 'objects', 'scene': 'scene'}

# --- ĐỊNH NGHĨA LOẠI NODE (Thêm Pseudo) ---
NODE_TYPES = {
    'scene': 0,
    'object': 1,
    'human': 2,
    'face': 3,
    'pseudo': 4  # <--- MỚI: Nút đại diện nhóm
}

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_feature(base_path, feature_type, split, label_name, stem):
    try:
        folder_name = f'features_{feature_type}_FULL_zip'
        key = FEATURE_KEYS[feature_type]
        file_path = base_path / folder_name / split / label_name / f"{stem}.npz"
        
        if not file_path.exists(): return np.empty((0, FEATURE_DIMS[feature_type]), dtype=np.float32)
        data = np.load(file_path)
        if key not in data: return np.empty((0, FEATURE_DIMS[feature_type]), dtype=np.float32)
        return data[key]
    except: return np.empty((0, FEATURE_DIMS[feature_type]), dtype=np.float32)

def build_single_graph_with_pseudo(paths, stem, label_id):
    # 1. Tải nguyên liệu
    face_feats = load_feature(paths['base'], 'face', paths['split'], paths['label'], stem)
    human_feats = load_feature(paths['base'], 'human', paths['split'], paths['label'], stem)
    object_feats = load_feature(paths['base'], 'object', paths['split'], paths['label'], stem)
    scene_feat = load_feature(paths['base'], 'scene', paths['split'], paths['label'], stem)

    # Giới hạn số lượng node để graph không quá nặng (Chuẩn paper)
    if face_feats.shape[0] > 16: face_feats = face_feats[:16]
    if human_feats.shape[0] > 16: human_feats = human_feats[:16]
    if object_feats.shape[0] > 16: object_feats = object_feats[:16]
    
    # Xử lý Scene (luôn có 1)
    if scene_feat.shape[0] == 0: scene_feat = np.zeros((1, 2048), dtype=np.float32)

    n_faces = face_feats.shape[0]
    n_humans = human_feats.shape[0]
    n_objects = object_feats.shape[0]
    n_scene = 1
    
    # Tổng số node thật
    total_real_nodes = n_scene + n_objects + n_humans + n_faces
    
    if total_real_nodes == 0: return None

    # Tổng số node trong graph (Thêm 1 cho Pseudo)
    total_nodes = total_real_nodes + 1 

    # 2. Tạo Ma trận đặc trưng (x)
    x = torch.zeros((total_nodes, 2048), dtype=torch.float32)
    node_type_ids = torch.full((total_nodes,), -1, dtype=torch.long)
    
    # --- BƯỚC A: ĐIỀN CÁC NODE THẬT (Từ index 1 trở đi) ---
    curr = 1 
    
    # Scene
    x[curr] = torch.from_numpy(scene_feat[0])
    node_type_ids[curr] = NODE_TYPES['scene']
    curr += 1
    
    # Objects
    if n_objects > 0:
        x[curr : curr + n_objects] = torch.from_numpy(object_feats)
        node_type_ids[curr : curr + n_objects] = NODE_TYPES['object']
        curr += n_objects
        
    # Humans
    if n_humans > 0:
        x[curr : curr + n_humans] = torch.from_numpy(human_feats)
        node_type_ids[curr : curr + n_humans] = NODE_TYPES['human']
        curr += n_humans
        
    # Faces (Padding 512 -> 2048)
    if n_faces > 0:
        x[curr : curr + n_faces, :512] = torch.from_numpy(face_feats)
        node_type_ids[curr : curr + n_faces] = NODE_TYPES['face']
        curr += n_faces

    # --- BƯỚC B: TẠO PSEUDO NODE (Tại index 0) ---
    # Công thức FANet: Pseudo Node = Trung bình cộng các node thành phần
    # Lấy toàn bộ các node thật (từ index 1 đến hết)
    real_nodes_features = x[1:] 
    pseudo_feature = torch.mean(real_nodes_features, dim=0)
    
    # Gán vào vị trí đầu tiên
    x[0] = pseudo_feature
    node_type_ids[0] = NODE_TYPES['pseudo']

    # 3. Tạo Cạnh (Edge Index)
    src = []
    dst = []
    
    # KẾT NỐI:
    # 1. Pseudo Node (0) <--> Tất cả Node thật (i)
    # 2. Các Node thật vẫn nối với nhau (Fully Connected)
    for i in range(total_nodes):
        for j in range(total_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
                
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor(label_id, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y, node_type=node_type_ids, stem=stem)

def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/graphs')
    args = parser.parse_args()
    
    base_path = Path(args.base_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_graphs = {'train': [], 'val': [], 'test': []}
    splits = ['train', 'val', 'test']
    
    for split in splits:
        # Dùng Face làm chuẩn để đếm file
        split_path_scan = base_path / 'features_face_FULL_zip' / split
        if not split_path_scan.exists(): continue
            
        for label_name in LABEL_MAP.keys():
            label_id = LABEL_MAP[label_name]
            label_path = split_path_scan / label_name
            if not label_path.exists(): continue
            
            logging.info(f"Dang xu ly: {split}/{label_name}...")
            image_stems = [f.stem for f in label_path.glob('*.npz')]
            
            for stem in tqdm(image_stems):
                # GỌI HÀM MỚI CÓ PSEUDO NODE
                graph = build_single_graph_with_pseudo({'base': base_path, 'split': split, 'label': label_name}, stem, label_id)
                if graph is not None:
                    all_graphs[split].append(graph)
    
    output_file = output_dir / 'graph_data_pseudo.pkl' # Lưu tên khác để phân biệt
    with open(output_file, 'wb') as f:
        pickle.dump(all_graphs, f)
    logging.info(f"Da xong! File luu tai: {output_file}")

if __name__ == '__main__':
    main()