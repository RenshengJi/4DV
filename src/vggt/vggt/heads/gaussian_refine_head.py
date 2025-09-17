# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, List
import numpy as np
import time
import logging

# Try to import scikit-learn for KD-tree, fallback to custom implementation
try:
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available, using custom spatial hash implementation")


class PointNetEncoder(nn.Module):
    """PointNet-style encoder for processing point clouds with features."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
            
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, input_dim] point features
        Returns:
            [N, output_dim] encoded features
        """
        return self.encoder(x)


class LocalSelfAttentionBlock(nn.Module):
    """Local self-attention block for point cloud feature refinement with O(N*k) complexity."""
    
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, k_neighbors: int = 20):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.k_neighbors = k_neighbors
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.norm1 = nn.LayerNorm(dim)
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
        self.scale = self.head_dim ** -0.5
        
    def _build_knn_graph(self, positions: torch.Tensor, k: int) -> torch.Tensor:
        """
        Build k-nearest neighbor graph using optimal algorithm selection.
        
        Args:
            positions: [N, 3] 3D positions
            k: number of nearest neighbors
            
        Returns:
            knn_indices: [N, k] indices of k nearest neighbors for each point
        """
        N = positions.shape[0]
        device = positions.device
        
        knn_start_time = time.time()
        
        # Choose optimal algorithm based on point cloud size and available libraries
        if N <= 1000:
            # Small: use full matrix (fastest for very small N)
            algorithm = "full_matrix"
            knn_indices = self._build_knn_graph_full(positions, k)
        elif N <= 500000 and HAS_SKLEARN:
            # Small-Large: use KD-tree with sklearn (O(N log N)) - expanded limit
            algorithm = "kdtree"
            knn_indices = self._build_knn_graph_kdtree(positions, k)
        elif N <= 100000:
            # Medium without sklearn: use spatial hashing  
            algorithm = "spatial_hash"
            knn_indices = self._build_knn_graph_spatial_hash(positions, k)
        else:
            # Very Large: use hierarchical k-NN (improved batched processing)
            algorithm = "hierarchical"
            knn_indices = self._build_knn_graph_hierarchical(positions, k)
        
        knn_time = time.time() - knn_start_time
        
        
        return knn_indices
    
    def _build_knn_graph_full(self, positions: torch.Tensor, k: int) -> torch.Tensor:
        """
        Original full k-NN graph construction for small point clouds.
        """
        N = positions.shape[0]
        
        # Compute pairwise distances
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [N, N, 3]
        distances = torch.sum(diff ** 2, dim=2)  # [N, N]
        
        # Find k+1 nearest neighbors (including self)
        k_actual = min(k + 1, N)
        _, knn_indices = torch.topk(distances, k_actual, dim=1, largest=False)  # [N, k+1]
        
        # Remove self-connection
        if k_actual > 1:
            knn_indices = knn_indices[:, 1:]  # [N, k]
        else:
            knn_indices = knn_indices.repeat(1, k)[:, :k]  # [N, k]
            
        return knn_indices
    
    def _build_knn_graph_batched(self, positions: torch.Tensor, k: int) -> torch.Tensor:
        """
        DEPRECATED: 旧的分批k-NN方法，时间复杂度仍为O(N²)
        
        此方法虽然节省显存，但时间复杂度仍然是O(N²)，性能差。
        现在已被hierarchical方法替代，仅作为最后的备选方案保留。
        """
        N = positions.shape[0]
        device = positions.device
        
        # Adaptive batch size based on available memory and point count
        # Aim for distance matrices no larger than ~1GB
        max_distance_memory_gb = 1.0
        bytes_per_element = 4  # float32
        max_elements = max_distance_memory_gb * (1024**3) / bytes_per_element
        
        # batch_size × N should not exceed max_elements
        batch_size = max(100, min(2000, int(max_elements / N)))
        
        knn_indices = torch.zeros(N, k, dtype=torch.long, device=device)
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch_positions = positions[start_idx:end_idx]  # [batch_size, 3]
            batch_size_actual = end_idx - start_idx
            
            # Compute distances from batch points to all points
            # [batch_size, 1, 3] - [1, N, 3] = [batch_size, N, 3]
            diff = batch_positions.unsqueeze(1) - positions.unsqueeze(0)  # [batch_size, N, 3]
            distances = torch.sum(diff ** 2, dim=2)  # [batch_size, N]
            
            # Find k+1 nearest neighbors for this batch
            k_actual = min(k + 1, N)
            _, batch_knn_indices = torch.topk(distances, k_actual, dim=1, largest=False)  # [batch_size, k+1]
            
            # Remove self-connections and store results
            for i in range(batch_size_actual):
                global_idx = start_idx + i
                neighbors = batch_knn_indices[i]  # [k+1]
                
                # Remove self-connection
                mask = neighbors != global_idx
                valid_neighbors = neighbors[mask]  # Remove self
                
                if len(valid_neighbors) >= k:
                    knn_indices[global_idx] = valid_neighbors[:k]
                else:
                    # If not enough valid neighbors, pad with repetitions
                    num_valid = len(valid_neighbors)
                    if num_valid > 0:
                        # Repeat the valid neighbors to fill k positions
                        repeated = valid_neighbors.repeat((k + num_valid - 1) // num_valid)[:k]
                        knn_indices[global_idx] = repeated
                    else:
                        # Fallback: use random indices (shouldn't happen in practice)
                        random_indices = torch.randperm(N, device=device)[:k]
                        knn_indices[global_idx] = random_indices
        
        return knn_indices
    
    def _build_knn_graph_kdtree(self, positions: torch.Tensor, k: int) -> torch.Tensor:
        """
        Build k-NN graph using scikit-learn's KD-tree implementation.
        Time complexity: O(N log N), much faster than O(N²) for large N.
        
        Args:
            positions: [N, 3] 3D positions
            k: number of nearest neighbors
            
        Returns:
            knn_indices: [N, k] indices of k nearest neighbors for each point
        """
        N = positions.shape[0]
        device = positions.device
        
        # Convert to numpy for sklearn
        positions_np = positions.detach().cpu().numpy()
        
        # Build KD-tree and find k+1 nearest neighbors (including self)
        # Using 'kd_tree' algorithm which is optimal for 3D data
        nbrs = NearestNeighbors(
            n_neighbors=min(k + 1, N),
            algorithm='kd_tree',
            metric='euclidean',
            n_jobs=1  # Single thread to avoid potential issues with PyTorch
        )
        nbrs.fit(positions_np)
        
        # Find neighbors
        distances, indices = nbrs.kneighbors(positions_np)
        
        # Remove self-connection (first column is always the point itself)
        if indices.shape[1] > 1:
            knn_indices = indices[:, 1:]  # [N, k]
        else:
            # Edge case: only one point or k=0
            knn_indices = np.tile(indices, (1, k))[:, :k]
        
        # Ensure we have exactly k neighbors per point
        if knn_indices.shape[1] < k:
            # Pad with repetitions if needed
            last_col = knn_indices[:, -1:]
            padding = np.tile(last_col, (1, k - knn_indices.shape[1]))
            knn_indices = np.concatenate([knn_indices, padding], axis=1)
        elif knn_indices.shape[1] > k:
            knn_indices = knn_indices[:, :k]
        
        # Convert back to torch tensor
        return torch.from_numpy(knn_indices).to(device, dtype=torch.long)
    
    def _build_knn_graph_spatial_hash(self, positions: torch.Tensor, k: int) -> torch.Tensor:
        """
        Use spatial hashing to accelerate k-NN search when sklearn is not available.
        Divides space into grid cells and searches within nearby cells.
        Time complexity: O(N) average case, O(N²) worst case.
        """
        N = positions.shape[0]
        device = positions.device
        
        # Determine grid size based on point density
        # Aim for roughly 50-200 points per cell for optimal performance
        target_points_per_cell = min(100, max(10, N // 100))
        num_cells_per_dim = max(1, int((N / target_points_per_cell) ** (1/3)))
        
        # Compute bounding box
        min_coords = torch.min(positions, dim=0)[0]  # [3]
        max_coords = torch.max(positions, dim=0)[0]  # [3]
        
        # Avoid division by zero
        coord_range = max_coords - min_coords
        coord_range = torch.where(coord_range < 1e-6, torch.ones_like(coord_range), coord_range)
        
        # Add small padding to avoid edge cases
        coord_range = coord_range * 1.001
        
        # Compute grid cell size
        cell_size = coord_range / num_cells_per_dim  # [3]
        
        # Assign points to grid cells
        cell_indices = ((positions - min_coords) / cell_size).long()  # [N, 3]
        cell_indices = torch.clamp(cell_indices, 0, num_cells_per_dim - 1)
        
        # Convert 3D cell indices to 1D for hashing
        cell_hash = (cell_indices[:, 0] * num_cells_per_dim * num_cells_per_dim + 
                    cell_indices[:, 1] * num_cells_per_dim + 
                    cell_indices[:, 2])  # [N]
        
        # Prepare result tensor
        knn_indices = torch.zeros(N, k, device=device, dtype=torch.long)
        
        # Process each point
        for point_idx in range(N):
            point_pos = positions[point_idx]  # [3]
            point_cell = cell_indices[point_idx]  # [3]
            
            # Determine search radius in cells (adaptive based on expected k)
            search_radius = max(1, min(3, int(np.ceil((k / target_points_per_cell) ** (1/3)))))
            
            # Collect candidate points from nearby cells
            candidates = []
            
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    for dz in range(-search_radius, search_radius + 1):
                        nx = point_cell[0] + dx
                        ny = point_cell[1] + dy
                        nz = point_cell[2] + dz
                        
                        # Check bounds
                        if (0 <= nx < num_cells_per_dim and 
                            0 <= ny < num_cells_per_dim and 
                            0 <= nz < num_cells_per_dim):
                            
                            neighbor_hash = (nx * num_cells_per_dim * num_cells_per_dim + 
                                           ny * num_cells_per_dim + nz)
                            
                            # Find points in this neighboring cell
                            mask = (cell_hash == neighbor_hash)
                            neighbor_points = torch.where(mask)[0]
                            candidates.extend(neighbor_points.tolist())
            
            if len(candidates) > 0:
                candidates = torch.tensor(candidates, device=device, dtype=torch.long)
                candidates = torch.unique(candidates)  # Remove duplicates
                
                # Remove self
                candidates = candidates[candidates != point_idx]
                
                if len(candidates) >= k:
                    # Compute distances to all candidates
                    candidate_pos = positions[candidates]  # [num_candidates, 3]
                    distances = torch.sum((point_pos - candidate_pos) ** 2, dim=1)  # [num_candidates]
                    
                    # Find k nearest
                    _, nearest_indices = torch.topk(distances, k, largest=False)
                    nearest_points = candidates[nearest_indices]
                    knn_indices[point_idx] = nearest_points
                else:
                    # Not enough neighbors in local cells, expand search
                    # Compute distances to all points (fallback)
                    all_distances = torch.sum((point_pos - positions) ** 2, dim=1)
                    all_distances[point_idx] = float('inf')  # Exclude self
                    _, nearest_indices = torch.topk(all_distances, k, largest=False)
                    knn_indices[point_idx] = nearest_indices
            else:
                # No candidates found (shouldn't happen), use random fallback
                random_indices = torch.randperm(N, device=device)[:k+1]
                random_indices = random_indices[random_indices != point_idx][:k]
                if len(random_indices) < k:
                    # Pad with repetitions
                    padding = torch.zeros(k - len(random_indices), device=device, dtype=torch.long)
                    random_indices = torch.cat([random_indices, padding])
                knn_indices[point_idx] = random_indices
        
        return knn_indices

    def _build_knn_graph_hierarchical(self, positions: torch.Tensor, k: int) -> torch.Tensor:
        """
        高效的分层k-NN算法，用于超大规模点云 (>500k points)
        
        策略：
        1. 空间分区：将点云分为多个空间区域
        2. 局部搜索：在每个区域内使用高效算法
        3. 边界处理：处理跨区域的邻居关系
        
        时间复杂度: O(N log N) 平均情况
        """
        N = positions.shape[0]
        device = positions.device
        
        # 如果sklearn可用且规模不是特别大，仍使用KD-tree
        if N <= 1000000 and HAS_SKLEARN:
            try:
                return self._build_knn_graph_kdtree(positions, k)
            except:
                pass  # Fallback to hierarchical approach
        
        # 计算点云的空间边界
        min_coords = torch.min(positions, dim=0)[0]  # [3]
        max_coords = torch.max(positions, dim=0)[0]  # [3]
        coord_range = max_coords - min_coords
        
        # 防止除零错误
        coord_range = torch.where(coord_range < 1e-6, torch.ones_like(coord_range), coord_range)
        
        # 自适应确定分区数量：目标每个分区包含10K-50K个点
        target_points_per_partition = min(50000, max(10000, N // 20))
        num_partitions_total = max(8, N // target_points_per_partition)
        
        # 3D分区：尽量均匀分布
        num_partitions_per_dim = max(2, int(np.ceil(num_partitions_total ** (1/3))))
        
        # 计算分区大小
        partition_size = coord_range / num_partitions_per_dim  # [3]
        
        # 为每个点分配分区ID
        partition_indices = ((positions - min_coords) / partition_size).long()  # [N, 3]
        partition_indices = torch.clamp(partition_indices, 0, num_partitions_per_dim - 1)
        
        # 转换为1D分区索引以便处理
        partition_ids = (partition_indices[:, 0] * num_partitions_per_dim * num_partitions_per_dim + 
                        partition_indices[:, 1] * num_partitions_per_dim + 
                        partition_indices[:, 2])  # [N]
        
        # 构建分区到点的映射
        unique_partitions = torch.unique(partition_ids)
        knn_indices = torch.zeros(N, k, dtype=torch.long, device=device)
        
        for partition_id in unique_partitions:
            # 获取当前分区内的点
            partition_mask = (partition_ids == partition_id)
            partition_point_indices = torch.nonzero(partition_mask, as_tuple=False).squeeze(1)
            partition_positions = positions[partition_point_indices]  # [P, 3]
            P = partition_positions.shape[0]
            
            if P <= 1:
                continue
                
            # 确定搜索邻居的范围：包括相邻分区
            current_3d_idx = torch.tensor([
                partition_id // (num_partitions_per_dim * num_partitions_per_dim),
                (partition_id // num_partitions_per_dim) % num_partitions_per_dim,  
                partition_id % num_partitions_per_dim
            ], dtype=torch.long)
            
            # 收集当前分区及其邻居分区的所有点
            neighbor_point_indices = []
            neighbor_positions = []
            
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        neighbor_3d_idx = current_3d_idx + torch.tensor([dx, dy, dz])
                        
                        # 检查边界
                        if torch.any(neighbor_3d_idx < 0) or torch.any(neighbor_3d_idx >= num_partitions_per_dim):
                            continue
                        
                        neighbor_partition_id = (neighbor_3d_idx[0] * num_partitions_per_dim * num_partitions_per_dim +
                                               neighbor_3d_idx[1] * num_partitions_per_dim + 
                                               neighbor_3d_idx[2])
                        
                        neighbor_mask = (partition_ids == neighbor_partition_id)
                        if torch.any(neighbor_mask):
                            neighbor_indices = torch.nonzero(neighbor_mask, as_tuple=False).squeeze(1)
                            neighbor_point_indices.append(neighbor_indices)
                            neighbor_positions.append(positions[neighbor_indices])
            
            # 合并所有邻居点
            if neighbor_point_indices:
                all_neighbor_indices = torch.cat(neighbor_point_indices, dim=0)
                all_neighbor_positions = torch.cat(neighbor_positions, dim=0)
            else:
                all_neighbor_indices = partition_point_indices
                all_neighbor_positions = partition_positions
            
            N_neighbors = all_neighbor_positions.shape[0]
            
            # 在局部区域内计算k-NN
            for i, global_idx in enumerate(partition_point_indices):
                query_pos = partition_positions[i]  # [3]
                
                # 计算到所有邻居的距离
                distances = torch.sum((query_pos.unsqueeze(0) - all_neighbor_positions) ** 2, dim=1)  # [N_neighbors]
                
                # 排除自己
                self_mask = (all_neighbor_indices != global_idx)
                valid_distances = distances[self_mask]
                valid_indices = all_neighbor_indices[self_mask]
                
                if len(valid_indices) == 0:
                    # 如果没有有效邻居，使用随机填充
                    knn_indices[global_idx] = torch.randint(0, N, (k,), device=device)
                    continue
                
                # 找到k个最近邻
                k_actual = min(k, len(valid_indices))
                _, topk_idx = torch.topk(valid_distances, k_actual, largest=False)
                selected_neighbors = valid_indices[topk_idx]
                
                # 填充结果
                if k_actual < k:
                    # 如果邻居不足k个，重复填充
                    padding_size = k - k_actual
                    if k_actual > 0:
                        padding_indices = selected_neighbors[torch.randint(0, k_actual, (padding_size,))]
                        selected_neighbors = torch.cat([selected_neighbors, padding_indices])
                    else:
                        selected_neighbors = torch.randint(0, N, (k,), device=device)
                
                knn_indices[global_idx] = selected_neighbors
        
        return knn_indices
        
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, dim] point features
            positions: [N, 3] 3D positions for building local neighborhoods
        Returns:
            [N, dim] refined features
        """
        N, dim = x.shape
        
        # If we have very few points, use regular attention
        if N <= self.k_neighbors:
            return self._full_attention_forward(x)
            
        # Additional memory check: if positions are not provided or invalid
        if positions is None or positions.shape[0] != N or positions.shape[1] != 3:
            # Fallback to full attention for safety
            if N <= 1000:  # Only for very small clouds
                return self._full_attention_forward(x)
            else:
                raise ValueError(f"Invalid positions tensor for local attention. Expected [{N}, 3], got {positions.shape if positions is not None else None}")
        
        # Start total inference timing
        infer_start_time = time.time()
        
        # Normalize input
        x_norm = self.norm1(x)
        
        # Build k-NN graph (timing is handled in _build_knn_graph)
        knn_indices = self._build_knn_graph(positions, self.k_neighbors)  # [N, k]
        
        # Project to Q, K, V
        Q = self.q_proj(x_norm)  # [N, dim]
        K = self.k_proj(x_norm)  # [N, dim]
        V = self.v_proj(x_norm)  # [N, dim]
        
        # Reshape for multi-head attention
        Q = Q.view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        K = K.view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        V = V.view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        
        # Gather neighbor features
        k = knn_indices.shape[1]
        neighbor_K = K[knn_indices]  # [N, k, num_heads, head_dim]
        neighbor_V = V[knn_indices]  # [N, k, num_heads, head_dim]
        
        # Compute local attention
        # Q: [N, num_heads, head_dim] -> [N, 1, num_heads, head_dim]
        # neighbor_K: [N, k, num_heads, head_dim]
        Q_expanded = Q.unsqueeze(1)  # [N, 1, num_heads, head_dim]
        
        # Attention scores: [N, 1, num_heads, head_dim] @ [N, k, num_heads, head_dim].transpose(-1, -2)
        # = [N, k, num_heads]
        attention_scores = torch.sum(Q_expanded * neighbor_K, dim=-1) * self.scale  # [N, k, num_heads]
        
        # Apply softmax
        attention_weights = torch.softmax(attention_scores, dim=1)  # [N, k, num_heads]
        
        # Weighted sum: [N, k, num_heads, 1] * [N, k, num_heads, head_dim] -> [N, num_heads, head_dim]
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # [N, k, num_heads, 1]
        attended_features = torch.sum(attention_weights_expanded * neighbor_V, dim=1)  # [N, num_heads, head_dim]
        
        # Reshape back
        attended_features = attended_features.view(N, dim)  # [N, dim]
        
        # Output projection
        attn_out = self.out_proj(attended_features)
        
        # Residual connection
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        
        return x
        
    def _full_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback to full attention for small point clouds."""
        x_norm = self.norm1(x)
        
        # Use standard multi-head attention
        Q = self.q_proj(x_norm)
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)
        
        N, dim = x.shape
        Q = Q.view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, N, head_dim]
        K = K.view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, N, head_dim]
        V = V.view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, N, head_dim]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [num_heads, N, N]
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [num_heads, N, N]
        attended_features = torch.matmul(attention_weights, V)  # [num_heads, N, head_dim]
        
        # Transpose back and reshape
        attended_features = attended_features.transpose(0, 1).reshape(N, dim)  # [N, dim]
        
        # Output projection
        attn_out = self.out_proj(attended_features)
        
        # Residual connection
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class SelfAttentionBlock(nn.Module):
    """Legacy self-attention block - kept for backward compatibility."""
    
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, dim] point features
        Returns:
            [N, dim] refined features
        """
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class GaussianRefineHead(nn.Module):
    """
    Gaussian参数细化网络头 - 使用局部注意力机制优化显存使用
    
    输入：聚合后的动态物体Gaussian参数
    输出：细化的Gaussian参数变化量（不包含velocity）
    
    优化特性：
    - 使用局部注意力将复杂度从O(N²)降低到O(N*k)
    - k=20的邻居数量平衡了性能和效果
    - 自动回退到完整注意力处理小规模点云
    """
    
    def __init__(
        self,
        input_gaussian_dim: int = 14,  # 原始Gaussian参数维度 
        output_gaussian_dim: int = 14,  # 输出参数维度(包含所有14维)
        feature_dim: int = 256,
        num_attention_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        k_neighbors: int = 20,  # 局部注意力的邻居数量
        use_local_attention: bool = True  # 是否使用局部注意力
    ):
        super().__init__()
        self.input_gaussian_dim = input_gaussian_dim
        self.output_gaussian_dim = output_gaussian_dim
        self.feature_dim = feature_dim
        self.k_neighbors = k_neighbors
        self.use_local_attention = use_local_attention
        
        # Optional timing logging flag for performance debugging
        self._log_timing = False
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_gaussian_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 选择注意力机制类型
        self.use_local_attention = use_local_attention
        self.k_neighbors = k_neighbors
        
        # Self-attention layers for feature refinement
        if use_local_attention:
            self.attention_layers = nn.ModuleList([
                LocalSelfAttentionBlock(feature_dim, num_heads, mlp_ratio, k_neighbors)
                for _ in range(num_attention_layers)
            ])
        else:
            # 保持原始的全局注意力机制以便对比
            self.attention_layers = nn.ModuleList([
                SelfAttentionBlock(feature_dim, num_heads, mlp_ratio)
                for _ in range(num_attention_layers)
            ])
        
        # Position encoding (learnable) - 优化显存使用
        # 降低预分配大小，并支持动态扩展
        max_prealloc_points = min(8192, 10000)  # 减少预分配大小
        self.pos_encoding = nn.Parameter(torch.randn(1, max_prealloc_points, feature_dim))
        self.max_prealloc_points = max_prealloc_points
        
        # Output projection for Gaussian parameter deltas
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, output_gaussian_dim)
        )
        
        # Initialize all layers with Xavier uniform
        self._init_weights()
        
    def _init_weights(self):
        """Initialize all linear layers with Xavier uniform, but output layer to zeros"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # **Critical modification**: Initialize the final output layer to zeros
        # This ensures the network initially outputs zero deltas, preserving original Gaussian parameters
        final_layer = self.output_proj[-1]  # The last Linear(128, 14) layer
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

        # Initialize position encoding with Xavier uniform
        nn.init.xavier_uniform_(self.pos_encoding.data.view(-1, self.pos_encoding.size(-1)))
        
    def forward(self, gaussian_params: torch.Tensor, point_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 - 优化了显存使用的局部注意力机制
        
        Args:
            gaussian_params: [N, input_gaussian_dim] 输入的Gaussian参数
            point_positions: [N, 3] 点的3D位置，用于构建局部邻域（局部注意力时必需）
            
        Returns:
            gaussian_deltas: [N, output_gaussian_dim] Gaussian参数的变化量
        """
        total_start_time = time.time()
        N = gaussian_params.shape[0]
        
        # 从Gaussian参数中提取3D位置作为默认位置信息
        if point_positions is None:
            # 假设Gaussian参数的前3维是位置信息
            point_positions = gaussian_params[:, :3]  # [N, 3]
        
        # Input projection
        features = self.input_proj(gaussian_params)  # [N, feature_dim]
        
        # Add position encoding - 优化显存使用
        if N <= self.max_prealloc_points:
            pos_enc = self.pos_encoding[:, :N, :]  # [1, N, feature_dim]
            features = features + pos_enc.squeeze(0)  # [N, feature_dim]
        else:
            # 对于超大规模点云，使用更节省显存的位置编码策略
            # 选择性插值而不是全量插值
            if N <= 50000:  # 中等规模：使用插值
                pos_enc = F.interpolate(
                    self.pos_encoding.permute(0, 2, 1),  # [1, feature_dim, max_prealloc]
                    size=N,
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)  # [1, N, feature_dim]
                features = features + pos_enc.squeeze(0)
            else:  # 超大规模：使用循环重复避免插值的显存开销
                # 重复使用现有的位置编码
                repeat_times = (N + self.max_prealloc_points - 1) // self.max_prealloc_points
                pos_enc_repeated = self.pos_encoding.repeat(1, repeat_times, 1)  # [1, repeat_times*max_prealloc, feature_dim]
                pos_enc = pos_enc_repeated[:, :N, :]  # [1, N, feature_dim]
                features = features + pos_enc.squeeze(0)
        
        # Apply self-attention layers
        for attention_layer in self.attention_layers:
            if self.use_local_attention:
                features = attention_layer(features, point_positions)  # 局部注意力需要位置信息
            else:
                features = attention_layer(features)  # 全局注意力
        
        # Output projection to get parameter deltas
        gaussian_deltas = self.output_proj(features)  # [N, output_gaussian_dim]
        
        # Apply activation functions to constrain delta values
        gaussian_deltas = self._apply_delta_activations(gaussian_deltas)
        
        
        return gaussian_deltas
    
    def _apply_delta_activations(self, deltas: torch.Tensor) -> torch.Tensor:
        """
        对delta输出应用激活函数，与cross_render_and_loss中完全一致
        
        Args:
            deltas: [N, 14] 原始delta输出
            
        Returns:
            activated_deltas: [N, 14] 应用激活函数后的delta
        """
        activated_deltas = deltas.clone()
        
        # 位置变化 (0:3) - 使用tanh抑制变化量
        activated_deltas[:, :3] = torch.tanh(deltas[:, :3]) * 0.01  # 限制在±0.1范围内
        
        # 尺度变化 (3:6) - 与cross_render_and_loss一致: (0.05 * torch.exp(scale)).clamp_max(0.3)
        activated_deltas[:, 3:6] = (0.05 * torch.exp(deltas[:, 3:6])).clamp_max(0.02)
        
        # 颜色变化 (6:9) - 不做特殊处理，保持原值
        activated_deltas[:, 6:9] = torch.tanh(deltas[:, 6:9]) * 0.1 # 限制在±0.1范围内

        # 旋转变化 (9:13) - 四元数归一化，与cross_render_and_loss一致
        rotations = deltas[:, 9:13]
        # Add safety checks for rotation normalization to prevent division by zero
        rotation_norms = torch.norm(rotations, dim=-1, keepdim=True)
        # Add small epsilon to prevent division by zero
        rotation_norms = torch.clamp(rotation_norms, min=1e-8)
        # 使用F.normalize来避免维度问题
        activated_deltas[:, 9:13] = F.normalize(rotations, p=2, dim=-1, eps=1e-8)
        
        # 透明度变化 (13:14) - 与cross_render_and_loss一致: sigmoid()
        activated_deltas[:, 13:14] = torch.tanh(deltas[:, 13:14]) * 0.5 # 限制在±0.5范围内
        
        return activated_deltas
    
    def apply_deltas(self, original_params: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """
        将预测的变化量应用到原始Gaussian参数上
        
        Args:
            original_params: [N, 14] 原始Gaussian参数
            deltas: [N, 14] 预测的变化量(已经过激活函数处理)
            
        Returns:
            refined_params: [N, 14] 细化后的Gaussian参数
        """
        # deltas已经过激活函数处理，其中：
        # - 位置(0:3): 原始delta值
        # - 尺度(3:6): 经过(0.05 * exp(x)).clamp_max(0.3)处理
        # - 颜色(6:9): 原始delta值  
        # - 旋转(9:13): 经过归一化处理的四元数
        # - 透明度(13:14): 经过sigmoid处理
        
        refined_params = original_params.clone()
        
        # 位置变化: 直接相加
        refined_params[:, :3] = original_params[:, :3] + deltas[:, :3]
        
        # 尺度变化: deltas已经是处理后的绝对值，不是增量
        # refined_params[:, 3:6] = original_params[:, 3:6] + deltas[:, 3:6]
        
        # 颜色变化: 直接相加
        refined_params[:, 6:9] = original_params[:, 6:9] + deltas[:, 6:9]
        
        # # 旋转变化: deltas已经是归一化的四元数，直接使用
        # refined_params[:, 9:13] = deltas[:, 9:13]  # 直接使用激活后的归一化四元数
        
        # # 透明度变化: deltas已经是sigmoid处理后的值，直接使用
        refined_params[:, 13:14] = original_params[:, 13:14] + deltas[:, 13:14]
        
        return refined_params