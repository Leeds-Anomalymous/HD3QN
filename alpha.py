import torch
import os

def print_alpha_values(save_dir):
    """打印所有模型文件中的 alpha 参数值"""
    
    dataset_name = 'TBM_0.01'
    rho = 0.01
    model_variant = 'Hierarchical_Transformer'
    reward_multiplier = 1
    high_discount_factor = 0.1
    low_discount_factor = 0.1
    high_eta = 0.05
    low_eta = 0.05
    ood_reward_scale = 0.5
    
    print(f"\n{'='*80}")
    print(f"打印模型的 alpha 参数值")
    print(f"{'='*80}\n")
    
    alpha_values = []
    
    for run in range(1, 6):  # 第1次到第5次
        model_filename = (
            f'{dataset_name}_rho{rho}_{model_variant}_reward{reward_multiplier}_'
            f'highGamma{high_discount_factor}_lowGamma{low_discount_factor}_'
            f'highEta{high_eta}_lowEta{low_eta}_oodScale{ood_reward_scale}_第{run}次.pth'
        )
        model_path = os.path.join(save_dir, model_filename)
        
        if not os.path.exists(model_path):
            print(f"❌ 第 {run} 次: 模型文件不存在 - {model_path}")
            continue
        
        try:
            # 加载模型
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 获取 low_q_net 的状态字典
            low_q_net_state = checkpoint['low_q_net']
            
            # 从 low_q_net 中提取 alpha 参数
            if 'transformer.alpha' in low_q_net_state:
                alpha_value = low_q_net_state['transformer.alpha'].item()
            else:
                alpha_value = None
            
            if alpha_value is not None:
                print(f"✓ 第 {run} 次: alpha = {alpha_value:.6f}")
                alpha_values.append(alpha_value)
            else:
                print(f"⚠ 第 {run} 次: 未找到 alpha 参数")
        
        except Exception as e:
            print(f"❌ 第 {run} 次: 加载模型出错 - {str(e)}")
    
    # 打印统计信息
    if alpha_values:
        print(f"\n{'='*80}")
        print(f"统计信息:")
        print(f"  平均值: {sum(alpha_values) / len(alpha_values):.6f}")
        print(f"  最小值: {min(alpha_values):.6f}")
        print(f"  最大值: {max(alpha_values):.6f}")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    # 修改为你的实际保存目录
    save_dir = 'D:\SWJT-Leeds\科创\RL\DRLimb-Multi\Good-past2'
    print_alpha_values(save_dir)