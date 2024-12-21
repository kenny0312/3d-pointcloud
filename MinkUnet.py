import torch
import MinkowskiEngine as ME
import torch.nn as nn
from IPython.core.magic_arguments import argument
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class HumanPointCloudDataset(Dataset):
    def __init__(self, data_dir, augment=False, compute_normals=False, k_neighbors=10):
       self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
       self.augment = augment  # 是否进行数据增强
       self.compute_normals = compute_normals  # 是否计算法向量
       self.k_neighbors = k_neighbors  # 用于计算法向量的邻居点数

    def __len__(self):
        return len(self.files)

    # def __getitem__(self, idx):
    #     file_path = self.files[idx]
    #     data = np.loadtxt(file_path, delimiter=",")
    #     coords = data[:, :3]  # 提取点坐标 (x, y, z)
    #     labels = data[:, -1].astype(np.int64)  # 提取标签
    #
    #     # 数据增强
    #     if self.augment:
    #         coords = self.augment_coords(coords)
    #
    #     # 计算法向量
    #     normals = None
    #     if self.compute_normals:
    #         normals = self.compute_normals_for_points(coords)
    #
    #     # 使用坐标和法向量作为特征
    #     features = coords if normals is None else np.hstack([coords, normals])
    #     return coords, features, labels
    def __getitem__(self, idx):
        file_path = self.files[idx]
        try:
            data = np.loadtxt(file_path, delimiter=",")
            assert data.shape[1] >= 4, f"File {file_path} must have at least 4 columns (x, y, z, label)"
        except Exception as e:
            raise ValueError(f"Error loading file {file_path}: {e}")

        coords = data[:, :3]  # 提取点坐标
        labels = data[:, -1].astype(np.int64)  # 提取标签
        features = coords  # 使用点坐标作为特征

        # 数据增强
        if self.augment:
            coords, labels, features = self.augment_coords(coords, labels, features)

        return coords, features, labels

    def augment_coords(self, coords, labels, features):
        indices = np.arange(coords.shape[0])

        # 示例：随机裁剪到 80%
        mask = np.random.choice(indices, size=int(0.8 * len(indices)), replace=False)
        coords = coords[mask]
        labels = labels[mask]  # 同步更新标签
        features = features[mask]
        return coords, labels, features

    def compute_normals_for_points(self, coords):
        normals = []

        for i, point in enumerate(coords):
            # 手动计算欧氏距离并获取 K 近邻索引
            distances = np.linalg.norm(coords - point, axis=1)
            neighbor_indices = np.argsort(distances)[:self.k_neighbors]

            # 获取近邻点并计算协方差矩阵
            neighbors = coords[neighbor_indices]
            neighbors_centered = neighbors - neighbors.mean(axis=0)
            cov_matrix = np.dot(neighbors_centered.T, neighbors_centered)

            # 计算最小特征值对应的特征向量（平面的法向量）
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            normal = eigenvectors[:, 0]  # 最小特征值对应的特征向量

            # 确保法向量一致指向外部
            if np.dot(normal, point - neighbors.mean(axis=0)) < 0:
                normal = -normal

            normals.append(normal)

        return np.array(normals)


class SimpleMinkUNet(nn.Module):
    def __init__(self, in_channels, out_channels, D=3):
        super().__init__()
        self.D = D
        self.inplanes = 16  # 缩小初始通道数
        self.conv0 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=3, stride=1, dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)
        self.relu = ME.MinkowskiReLU(inplace=True)

        # 定义简单的 Encoder
        self.conv1 = ME.MinkowskiConvolution(
            self.inplanes, 32, kernel_size=3, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(32)

        self.conv2 = ME.MinkowskiConvolution(
            32, 64, kernel_size=3, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(64)

        # Decoder
        self.convtr1 = ME.MinkowskiConvolutionTranspose(
            64, 32, kernel_size=3, stride=2, dimension=D)
        self.bntr1 = ME.MinkowskiBatchNorm(32)

        self.convtr2 = ME.MinkowskiConvolutionTranspose(
            32, out_channels, kernel_size=3, stride=2, dimension=D)

    def forward(self, x):
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bntr1(self.convtr1(x)))
        x = self.convtr2(x)
        return x


def collate_fn(batch):
    coords, features, labels = zip(*batch)
    batched_coords = np.vstack(coords)
    batched_features = np.vstack(features)
    batched_labels = np.hstack(labels)
    return batched_coords, batched_features, batched_labels

# 训练代码
def train_model(data_dir, num_epochs=10, batch_size=1, lr=0.01):
    dataset = HumanPointCloudDataset(data_dir, augment = True, compute_normals= True , k_neighbors=10)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)



    # 初始化简化模型
    model = SimpleMinkUNet(in_channels=3, out_channels=14, D=3).to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)


    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for coords, features, labels in train_loader:
            coords = ME.utils.batched_coordinates([coords])
            features = torch.tensor(features, dtype=torch.float32).to('cuda')
            labels = torch.tensor(labels, dtype=torch.long).to('cuda')
            input_tensor = ME.SparseTensor(features, coords, device='cuda')

            optimizer.zero_grad()
            outputs = model(input_tensor)
            loss = criterion(outputs.F, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    # 保存模型
    torch.save(model.state_dict(), "simplified_human_segmentation.pth")
    print("Model saved!")

def test_model(model_path, test_data_dir, batch_size=1):
    # 加载测试集
    test_dataset = HumanPointCloudDataset(test_data_dir, augment = False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 初始化模型并加载权重
    model = SimpleMinkUNet(in_channels=3, out_channels=14, D=3).to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total_points = 0

    with torch.no_grad():
        for coords, features, labels in test_loader:
            coords = ME.utils.batched_coordinates([coords])
            features = torch.tensor(features, dtype=torch.float32).to('cuda')
            labels = torch.tensor(labels, dtype=torch.long).to('cuda')
            input_tensor = ME.SparseTensor(features, coords, device='cuda')

            outputs = model(input_tensor)
            loss = criterion(outputs.F, labels)
            total_loss += loss.item()

            # 计算准确率
            preds = outputs.F.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_points += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total_points
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")


def test_model_and_save_results(model_path, test_data_dir, output_dir, num_files=5, batch_size=1):
# 加载测试集
    test_dataset = HumanPointCloudDataset(test_data_dir, augment = False, compute_normals= False , k_neighbors=10)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    selected_indices = random.sample(range(len(test_dataset)), num_files)

# 初始化模型并加载权重
    model = SimpleMinkUNet(in_channels=3, out_channels=14, D=3).to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

# 准备统计测试准确率的变量
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total_points = 0
    saved_files_count = 0

    with torch.no_grad():
         for idx, (coords, features, labels) in enumerate(test_loader):
           coords = ME.utils.batched_coordinates([coords])
           features = torch.tensor(features, dtype=torch.float32).to('cuda')
           labels = torch.tensor(labels, dtype=torch.long).to('cuda')
           input_tensor = ME.SparseTensor(features, coords, device='cuda')

        # 模型预测
           outputs = model(input_tensor)
           loss = criterion(outputs.F, labels)
           total_loss += loss.item()

        # 计算准确率
           preds = outputs.F.argmax(dim=1)
           correct += (preds == labels).sum().item()
           total_points += labels.size(0)

        # 如果当前批次对应要保存的索引，保存结果
           if idx in selected_indices and saved_files_count < num_files:
             coords_np = coords[:, 1:].cpu().numpy()  # 提取xyz坐标
             rgb = features.cpu().numpy()  # 提取RGB数据
             preds_np = preds.cpu().numpy()  # 获取预测结果
             result = np.hstack([coords_np, rgb, preds_np[:, None]])

            # 提取原始文件名
             original_file_name = os.path.basename(test_dataset.files[idx])
             output_file = os.path.join(output_dir, f"result_{os.path.splitext(original_file_name)[0]}.txt")

            # 保存到文件
             np.savetxt(output_file, result, fmt="%.6f", delimiter=",", header="x,y,z,r,g,b,prediction", comments='')
             print(f"Saved: {output_file} (from {original_file_name})")
             saved_files_count += 1

        # 计算整体测试结果
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total_points
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

# 主函数"C:\Users\26796\OneDrive\Desktop\dataset\testdata\train"
if __name__ == "__main__":
    data_dir = "/app/train"   # 替换为实际数据路径
    test_data_dir = "/app/test"
    train_model(data_dir)

    test_model("simplified_human_segmentation.pth", test_data_dir)
    output_dir = "/app/output_results"
    test_model_and_save_results(model_path, test_data_dir, output_dir)
