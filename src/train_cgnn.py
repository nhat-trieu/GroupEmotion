import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import pickle
import logging
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- Định nghĩa "Bộ não" C-GNN ---
# Đây là mô hình GNN sẽ "ăn" cái "nồi lẩu" của bạn

class CGNN_Model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        super(CGNN_Model, self).__init__()
        torch.manual_seed(42)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 1. "Sơ chế" (Projection Layer)
        # "Nồi lẩu" của bạn có "ADN" 2048-chiều (do ta đệm 0 cho face)
        # Lớp này sẽ "băm" 2048-chiều xuống còn "hidden_dim" (ví dụ: 512)
        self.projection = Linear(input_dim, hidden_dim)
        
        # 2. "Hầm" (GCN Layers)
        # Tạo ra 4 lớp "hầm" GNN
        self.conv_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            
        # 3. "Nêm nếm" (Classifier)
        # Sau khi "hầm", gộp (pool) tất cả "ADN" lại
        # và đưa qua 1 lớp "quyết định" (classifier)
        self.classifier = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. "Sơ chế"
        # x shape: [Tổng số node trong batch, 2048]
        x = self.projection(x)
        x = F.relu(x)
        
        # 2. "Hầm" 4 lần
        # Cho "ADN" chạy qua 4 lớp GNN
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            
        # 3. "Nêm nếm"
        # gộp (pool) tất cả node của TỪNG graph lại thành 1 vector
        # x shape: [batch_size, hidden_dim]
        x = global_mean_pool(x, batch)
        
        # Quyết định xem đây là Positive(2), Negative(0) hay Neutral(1)
        # out shape: [batch_size, 3]
        out = self.classifier(x)
        
        return out

# --- Thiết lập "Bữa tiệc" (Hàm Train/Eval) ---

def setup_logging(log_path):
    """Cài đặt logging để lưu lại nhật ký"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("Bat dau phien huan luyen (Training Session)...")

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(data.y.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    
    return acc, f1, precision, recall

# --- "Bắt đầu Ăn" (Hàm Main) ---

def main():
    parser = argparse.ArgumentParser(description='Train C-GNN Model')
    parser.add_argument('--graph_file', type=str, required=True, help='Duong dan den file graph_data_v1.pkl')
    parser.add_argument('--out_dir', type=str, default='/kaggle/working/checkpoint_cgnn', help='Noi luu model va log')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001) # 1e-4
    parser.add_argument('--hidden_dim', type=int, default=512, help='Kich thuoc "ADN" sau khi "bam"')
    
    args = parser.parse_args()
    
    # 1. Tạo thư mục output
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, 'training.log'))
    
    logging.info(f"Cac tham so: {args}")
    
    # 2. Cài đặt "nóc nhà"
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Dang su dung thiet bi: {device}")
    
    # 3. Mở "Kho Lẩu"
    logging.info(f"Dang tai 'kho lau' tu: {args.graph_file}")
    with open(args.graph_file, 'rb') as f:
        all_graphs = pickle.load(f)
        
    train_graphs = all_graphs['train']
    val_graphs = all_graphs['val']
    test_graphs = all_graphs['test']
    
    logging.info(f"So luong train/val/test: {len(train_graphs)} / {len(val_graphs)} / {len(test_graphs)}")
    
    # 4. Tạo "Dây chuyền" (DataLoaders)
    # (Tự động gộp 32 "nồi lẩu" lại thành 1 "lẩu thập cẩm" lớn)
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
    
    # 5. Thuê "Người nếm"
    model = CGNN_Model(
        input_dim=2048, # Vi ta da dem 0 (pad) tat ca len 2048
        hidden_dim=args.hidden_dim,
        output_dim=3 # (Negative, Neutral, Positive)
    ).to(device)
    
    logging.info(f"Da khoi tao model:\n{model}")
    
    # 6. Mua "Roi vọt" (Loss) và "Kẹo" (Optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 7. Bắt đầu "Dạy"
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        
        logging.info(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
        
        # Nếu "người nếm" học giỏi hơn, lưu "bộ não" nó lại
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.out_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'--- Luu model moi tot nhat tai {best_model_path} (Val Acc: {val_acc:.4f}) ---')

    logging.info("--- HET GIO (DA DAY XONG)! ---")
    
    # 8. "Thi Tốt nghiệp" (Test)
    logging.info("--- Dang cham diem 'thi tot nghiep' (Test set)... ---")
    
    # Tải "bộ não" xịn nhất (best_model.pt)
    model.load_state_dict(torch.load(best_model_path))
    
    test_acc, test_f1, test_precision, test_recall = evaluate(model, test_loader, device)
    
    logging.info("="*50)
    logging.info(f"KET QUA THI TOT NGHIEP (TEST SET)")
    logging.info(f"Accuracy: {test_acc:.4f}")
    logging.info(f"F1-score (Macro): {test_f1:.4f}")
    logging.info(f"Precision (Macro): {test_precision:.4f}")
    logging.info(f"Recall (Macro): {test_recall:.4f}")
    logging.info("="*50)
    logging.info(f"Model tot nhat (tam bang tot nghiep) da duoc luu tai: {best_model_path}")

if __name__ == '__main__':
    main()