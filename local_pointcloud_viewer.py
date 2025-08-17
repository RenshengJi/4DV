#!/usr/bin/env python3
"""
本地点云查看器
用于在本地计算机上交互式查看保存的点云文件
支持多种格式：PLY, PCD, OBJ
"""

import os
import numpy as np
import open3d as o3d
import argparse
import glob
from typing import List, Optional
import tkinter as tk
from tkinter import filedialog, messagebox
import threading


class LocalPointCloudViewer:
    """本地点云查看器"""
    
    def __init__(self, pointcloud_dir: str = "./saved_pointclouds"):
        """
        初始化查看器
        
        Args:
            pointcloud_dir: 点云文件目录
        """
        self.pointcloud_dir = pointcloud_dir
        self.current_index = 0
        self.point_clouds = []
        self.file_paths = []
        self.names = []
        self.vis = None
        
    def load_point_clouds_from_directory(self, directory: str) -> bool:
        """
        从目录加载点云文件
        
        Args:
            directory: 点云文件目录
            
        Returns:
            是否成功加载
        """
        self.point_clouds = []
        self.file_paths = []
        self.names = []
        
        # 支持的格式
        extensions = ['*.ply', '*.pcd', '*.obj']
        
        # 查找所有点云文件
        pointcloud_files = []
        for ext in extensions:
            pointcloud_files.extend(glob.glob(os.path.join(directory, ext)))
        
        if not pointcloud_files:
            print(f"在目录 {directory} 中没有找到点云文件")
            return False
        
        print(f"找到 {len(pointcloud_files)} 个点云文件")
        
        # 加载每个点云文件
        for file_path in sorted(pointcloud_files):
            try:
                print(f"加载点云文件: {file_path}")
                pcd = o3d.io.read_point_cloud(file_path)
                
                if len(pcd.points) == 0:
                    print(f"警告: 文件 {file_path} 为空，跳过")
                    continue
                
                self.point_clouds.append(pcd)
                self.file_paths.append(file_path)
                self.names.append(os.path.basename(file_path))
                
                print(f"成功加载: {os.path.basename(file_path)} (点数: {len(pcd.points)})")
                
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")
                continue
        
        return len(self.point_clouds) > 0
    
    def setup_visualization(self):
        """设置可视化环境"""
        if not self.point_clouds:
            print("没有点云可以显示")
            return False
        
        # 创建可视化窗口
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("本地点云查看器", width=1400, height=900)
        
        # 设置渲染选项
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # 深灰色背景
        opt.point_size = 2.0  # 点大小
        opt.show_coordinate_frame = True  # 显示坐标轴
        opt.line_width = 2.0
        
        # 添加坐标轴
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(coordinate_frame)
        
        return True
    
    def display_current_pointcloud(self):
        """显示当前点云"""
        if not self.point_clouds or self.vis is None:
            return
        
        # 清除之前的点云
        for geometry in self.vis.get_geometry():
            if isinstance(geometry, o3d.geometry.PointCloud):
                self.vis.remove_geometry(geometry, False)
        
        # 显示当前点云
        current_pcd = self.point_clouds[self.current_index]
        self.vis.add_geometry(current_pcd)
        
        # 更新窗口标题
        window_title = f"本地点云查看器 - {self.names[self.current_index]} ({self.current_index + 1}/{len(self.point_clouds)})"
        self.vis.get_window_name = lambda: window_title
        
        print(f"当前显示: {self.names[self.current_index]} (点数: {len(current_pcd.points)})")
    
    def next_pointcloud(self):
        """显示下一个点云"""
        if self.point_clouds:
            self.current_index = (self.current_index + 1) % len(self.point_clouds)
            self.display_current_pointcloud()
    
    def previous_pointcloud(self):
        """显示上一个点云"""
        if self.point_clouds:
            self.current_index = (self.current_index - 1) % len(self.point_clouds)
            self.display_current_pointcloud()
    
    def jump_to_pointcloud(self, index: int):
        """跳转到指定点云"""
        if 0 <= index < len(self.point_clouds):
            self.current_index = index
            self.display_current_pointcloud()
    
    def save_screenshot(self, filename: str = None):
        """保存截图"""
        if self.vis is None:
            return
        
        if filename is None:
            filename = f"screenshot_{self.names[self.current_index].split('.')[0]}.png"
        
        self.vis.capture_screen_image(filename)
        print(f"截图已保存: {filename}")
    
    def get_pointcloud_info(self) -> dict:
        """获取当前点云信息"""
        if not self.point_clouds:
            return {}
        
        pcd = self.point_clouds[self.current_index]
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        info = {
            'name': self.names[self.current_index],
            'file_path': self.file_paths[self.current_index],
            'point_count': len(pcd.points),
            'has_colors': pcd.has_colors(),
            'has_normals': pcd.has_normals(),
            'bounds': {
                'x_min': float(np.min(points[:, 0])),
                'x_max': float(np.max(points[:, 0])),
                'y_min': float(np.min(points[:, 1])),
                'y_max': float(np.max(points[:, 1])),
                'z_min': float(np.min(points[:, 2])),
                'z_max': float(np.max(points[:, 2]))
            }
        }
        
        if colors is not None:
            info['color_stats'] = {
                'r_mean': float(np.mean(colors[:, 0])),
                'g_mean': float(np.mean(colors[:, 1])),
                'b_mean': float(np.mean(colors[:, 2]))
            }
        
        return info
    
    def run_interactive(self):
        """运行交互式可视化"""
        if not self.setup_visualization():
            return
        
        # 显示第一个点云
        self.display_current_pointcloud()
        
        print("\n=== 本地点云查看器 ===")
        print("控制说明:")
        print("  鼠标左键: 旋转视角")
        print("  鼠标右键: 平移视角")
        print("  鼠标滚轮: 缩放")
        print("  键盘控制:")
        print("    K/k: 下一个点云")
        print("    J/j: 上一个点云")
        print("    数字键 1-9: 跳转到对应点云")
        print("    S/s: 保存截图")
        print("    I/i: 显示点云信息")
        print("    Q/q: 退出")
        print("=" * 30)
        
        while True:
            self.vis.poll_events()
            self.vis.update_renderer()
            
            # 检查键盘输入
            key = self.vis.poll_events()
            
            if key == ord('K') or key == ord('k'):
                self.next_pointcloud()
                
            elif key == ord('J') or key == ord('j'):
                self.previous_pointcloud()
                
            elif key == ord('S') or key == ord('s'):
                self.save_screenshot()
                
            elif key == ord('I') or key == ord('i'):
                info = self.get_pointcloud_info()
                print("\n当前点云信息:")
                for key, value in info.items():
                    if key == 'bounds':
                        print(f"  边界: X[{value['x_min']:.3f}, {value['x_max']:.3f}] "
                              f"Y[{value['y_min']:.3f}, {value['y_max']:.3f}] "
                              f"Z[{value['z_min']:.3f}, {value['z_max']:.3f}]")
                    elif key == 'color_stats':
                        print(f"  颜色统计: R={value['r_mean']:.3f} G={value['g_mean']:.3f} B={value['b_mean']:.3f}")
                    else:
                        print(f"  {key}: {value}")
                
            elif key == ord('Q') or key == ord('q'):
                break
                
            # 数字键跳转
            elif ord('1') <= key <= ord('9'):
                index = key - ord('1')
                if index < len(self.point_clouds):
                    self.jump_to_pointcloud(index)
        
        self.vis.destroy_window()


class PointCloudViewerGUI:
    """点云查看器GUI界面"""
    
    def __init__(self):
        """初始化GUI"""
        self.root = tk.Tk()
        self.root.title("本地点云查看器")
        self.root.geometry("400x300")
        
        self.viewer = None
        self.setup_gui()
    
    def setup_gui(self):
        """设置GUI界面"""
        # 主框架
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = tk.Label(main_frame, text="本地点云查看器", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 选择目录按钮
        select_btn = tk.Button(main_frame, text="选择点云目录", 
                              command=self.select_directory, 
                              width=20, height=2)
        select_btn.pack(pady=10)
        
        # 当前目录显示
        self.dir_label = tk.Label(main_frame, text="未选择目录", 
                                 wraplength=350, fg="gray")
        self.dir_label.pack(pady=10)
        
        # 文件信息显示
        self.info_label = tk.Label(main_frame, text="", wraplength=350)
        self.info_label.pack(pady=10)
        
        # 开始查看按钮
        self.view_btn = tk.Button(main_frame, text="开始查看", 
                                 command=self.start_viewer, 
                                 width=20, height=2, state=tk.DISABLED)
        self.view_btn.pack(pady=10)
        
        # 退出按钮
        exit_btn = tk.Button(main_frame, text="退出", 
                            command=self.root.quit, 
                            width=20, height=2)
        exit_btn.pack(pady=10)
    
    def select_directory(self):
        """选择点云目录"""
        directory = filedialog.askdirectory(title="选择点云文件目录")
        if directory:
            self.dir_label.config(text=f"已选择: {directory}")
            
            # 检查目录中的点云文件
            viewer = LocalPointCloudViewer(directory)
            if viewer.load_point_clouds_from_directory(directory):
                self.viewer = viewer
                self.info_label.config(text=f"找到 {len(viewer.point_clouds)} 个点云文件")
                self.view_btn.config(state=tk.NORMAL)
            else:
                self.info_label.config(text="未找到点云文件")
                self.view_btn.config(state=tk.DISABLED)
    
    def start_viewer(self):
        """启动查看器"""
        if self.viewer is None:
            messagebox.showerror("错误", "请先选择包含点云文件的目录")
            return
        
        # 在新线程中启动查看器
        def run_viewer():
            try:
                self.viewer.run_interactive()
            except Exception as e:
                messagebox.showerror("错误", f"启动查看器时出错: {e}")
        
        thread = threading.Thread(target=run_viewer)
        thread.daemon = True
        thread.start()
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="本地点云查看器")
    parser.add_argument("--pointcloud_dir", type=str, default="./saved_pointclouds",
                       help="点云文件目录")
    parser.add_argument("--gui", action="store_true",
                       help="启动GUI界面")
    parser.add_argument("--auto_load", action="store_true",
                       help="自动加载指定目录")
    
    args = parser.parse_args()
    
    if args.gui:
        # 启动GUI界面
        app = PointCloudViewerGUI()
        app.run()
    else:
        # 命令行模式
        viewer = LocalPointCloudViewer(args.pointcloud_dir)
        
        if args.auto_load:
            # 自动加载指定目录
            if not viewer.load_point_clouds_from_directory(args.pointcloud_dir):
                print(f"无法从目录 {args.pointcloud_dir} 加载点云文件")
                return
        else:
            # 手动选择目录
            print("请选择包含点云文件的目录:")
            directory = input("目录路径 (回车使用默认): ").strip()
            if not directory:
                directory = args.pointcloud_dir
            
            if not viewer.load_point_clouds_from_directory(directory):
                print(f"无法从目录 {directory} 加载点云文件")
                return
        
        # 启动交互式查看器
        viewer.run_interactive()


if __name__ == "__main__":
    main() 