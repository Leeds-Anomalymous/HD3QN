import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def plot_loss_comparison():
    """
    读取两个损失历史npy文件并使用subfigure绘制对比图
    """
    # 设置保存目录路径
    save_dir1 = '/workspace/RL/DQNimb/discount_results'
    save_dir2 = '/workspace/RL/DQNimb/loss_compare_results'
    
    # 定义要读取的文件名
    file1 = 'TBM_K_M_Noise_rho0.005_TBM_conv1d_4layer_reward1_gamma0.01_训练完成比1_第8次_loss_history.npy'
    file2 = 'TBM_K_M_Noise_rho0.005_TBM_conv1d_4layer_reward1_gamma0.9_训练完成比1_第2次_loss_history.npy'
    
    # 构建完整文件路径
    file1_path = os.path.join(save_dir1, file1)
    file2_path = os.path.join(save_dir2, file2)
    
    try:
        # 读取npy文件
        loss_history1 = np.load(file1_path)
        loss_history2 = np.load(file2_path)
        
        print(f"成功读取文件1: {file1}")
        print(f"损失历史长度: {len(loss_history1)}")
        print(f"成功读取文件2: {file2}")
        print(f"损失历史长度: {len(loss_history2)}")
        
        # 创建包含两个子图的figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))  # 修改为2行1列
        
        # 绘制第一个损失曲线
        ax1.plot(loss_history1, linewidth=1.5)
        ax1.set_title(r'(a) Discount Factor $\gamma = 0.1$', fontsize=18)
        ax1.set_xlabel('Training steps', fontsize=18)
        ax1.set_ylabel('Loss', fontsize=18)
        #ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=18)
        
        # 绘制第二个损失曲线
        ax2.plot(loss_history2, linewidth=1.5)
        ax2.set_title(r'(b) Discount Factor $\gamma = 0.9$', fontsize=18)
        ax2.set_xlabel('Training steps', fontsize=18)
        ax2.set_ylabel('Loss', fontsize=18)
        #ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=18)
        
        # 设置总标题
        #fig.suptitle('Loss Comparison: TBM_K_M_Noise (rho=0.005, TBM_conv1d_4layer)', fontsize=16)
        
        # 调整子图之间的间距
        plt.tight_layout()
        
        # 保存图片
        out_dir = '/workspace/RL/DQNimb/chores'
        output_filename = 'loss_comparison_gamma_0.1_vs_0.9.png'
        output_path = os.path.join(out_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"对比图已保存到: {output_path}")
        
    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
        print("请检查文件路径和文件名是否正确")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    plot_loss_comparison()
