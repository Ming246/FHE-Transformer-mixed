import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import warnings

def nexus_exp(x: np.ndarray,r:int) -> np.ndarray:
    exp_x= 1+x/(2**r)
    for _ in range (r):
        exp_x= exp_x**2
    return exp_x

def thor_exp(x: np.ndarray,delta:int,b:float) :
    x = x/8/delta
    p = np.array([ 0.032855468333339584, 0.05948672763856172, 0.03881607331549499, 0.0670090353368128, 
            0.15202099984697098, 0.20618261949210986, 0.23721029007596767, 0.26787311936472025, 
            0.27220647178765545, 0.2379982262906916, 0.1780344447042791, 0.11128698173597897, 
            0.05566510463488879, 0.020873931555133732, 0.005218196900295354, 0.0006522770224130905] )
    p=p*1533


    exp_approx = np.zeros_like(x)
    for coeff in p:
        exp_approx = exp_approx * x + coeff

    for _ in range(int(np.log2(delta))):    
        exp_approx = exp_approx**2
    return exp_approx

def goldschmidt_inverse(x: float, iterations: int) -> float:
    # 检查输入范围
    if x <= 0 or x >= 2:
        print(f"警告: 输入x={x}，算法在(0,2)范围外可能不收敛")
    
    # 步骤1: 初始化
    y = 1.0 - x        # 计算 (1-x)
    result = 2.0 - x   # 初始结果 = 1 + (1-x) = 2-x
    # 步骤2: 迭代计算
    for i in range(iterations):    # 每次迭代乘法深度加2
        y = y * y                  # 计算 (1-x)^(2^i)
        tmp = 1.0 + y              # 计算 1 + (1-x)^(2^i)
        result = result * tmp      # 累乘
        
        # 计算当前误差
        exact_value = 1.0 / x
        error = abs(result - exact_value)
        rel_error = error / exact_value * 100
        
    
    return result

def adaptive_goldschmidt_inverse(
    numerator: float, 
    denominator: float, 
    epsilon: float = 2**(-11), 
    alpha : float = 0.1
) -> Tuple[float, float]:
    num =0
    # 初始化变量
    a = numerator      # 对应an，倒数的近似值
    b = denominator    # 对应bn，分母
    e = epsilon        # 误差估计

    # 迭代循环
    while e < 1 - alpha:
        k = 2 / (e + 1)
        b_temp = 2 - k * b
        b = k * b * b_temp
        a = k * a * b_temp
        e_temp = 2 - k * e
        e = k * e * e_temp

        num+=1
    print(f"此次迭代 {num} 次")

    return a, e

def standard_softmax(x: np.ndarray) -> np.ndarray:
    """标准Softmax实现（作为基准）"""
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

def nexus_softmax(x: np.ndarray,r:int,iterations:int,scale:float) -> np.ndarray:
    exp_x = nexus_exp(x,r) 
    exp_x = exp_x/scale
    sum = np.sum(exp_x)
    print(f"nexus倒数和：{sum}")
    div_sum = goldschmidt_inverse(sum,iterations)
    return exp_x * div_sum
    #return exp_x / sum

def thor_softmax(
    x: np.ndarray,
    input_range: Tuple[float, float] ,
    delta1: int ,
    delta2: int 
) -> np.ndarray:

    mid_point = (input_range[0] + input_range[1]) / 2
    b = input_range[1] - mid_point

    x = x - mid_point
    x_scaled = x / delta1/delta2/8
    p = np.array([ 0.032855468333339584, 0.05948672763856172, 0.03881607331549499, 0.0670090353368128, 
            0.15202099984697098, 0.20618261949210986, 0.23721029007596767, 0.26787311936472025, 
            0.27220647178765545, 0.2379982262906916, 0.1780344447042791, 0.11128698173597897, 
            0.05566510463488879, 0.020873931555133732, 0.005218196900295354, 0.0006522770224130905] )
    p=p*1533
    en = 2**(-11)

    exp_approx = np.zeros_like(x_scaled)

    for coeff in p:
        exp_approx = exp_approx * x_scaled + coeff
    exp_approx = exp_approx/(np.exp(b/delta2/delta1))
    for _ in range(int(np.log2(delta1))):
        exp_approx = exp_approx ** 2  
    exp_approx = exp_approx/16
    sigma_exp = np.sum(exp_approx)
    print(f"thor倒数和：{sigma_exp}")
    inv_sigma,en = adaptive_goldschmidt_inverse(1,sigma_exp, epsilon=en, alpha=0.1/10)
    #inv_sigma = goldschmidt_inverse(sigma_exp,8)
    y = exp_approx * inv_sigma
    for _ in range(int(np.log2(delta2))):

        y_squared = y ** 2
        sum_y_sq = np.sum(y_squared)
        en = en/128/2
        print(f"thor倒数和：{sum_y_sq}")
        inv_sum_sq,en = adaptive_goldschmidt_inverse(1,sum_y_sq, epsilon=en, alpha=0.1/10)
        #inv_sum_sq = goldschmidt_inverse(sum_y_sq,6)
        y = y_squared * inv_sum_sq
    return y

def my_softmax(x: np.ndarray,delta:int,b:float,iterations:int,scale:float =1) -> np.ndarray:
    x = x/8/delta
    p = np.array([ 0.032855468333339584, 0.05948672763856172, 0.03881607331549499, 0.0670090353368128, 
            0.15202099984697098, 0.20618261949210986, 0.23721029007596767, 0.26787311936472025, 
            0.27220647178765545, 0.2379982262906916, 0.1780344447042791, 0.11128698173597897, 
            0.05566510463488879, 0.020873931555133732, 0.005218196900295354, 0.0006522770224130905] )
    p=p*1533


    exp_approx = np.zeros_like(x)
    for coeff in p:
        exp_approx = exp_approx * x + coeff
    # real_exp = np.exp(x/delta)
    # print(np.mean(real_exp / exp_approx))
    exp_approx = exp_approx/(np.exp(b/delta))
    for _ in range(int(np.log2(delta))):    
        exp_approx = exp_approx**2
    exp_approx = exp_approx/16
    #exp_approx = thor_exp(x,delta,b)
    sum = np.sum(exp_approx)
    
    print(f"my倒数和：{sum}")
    # print(f"real range:({np.min(exp_approx)},{np.max(exp_approx)})")
    # print(f"倒数和：{sum}")
    #div_sum,en = adaptive_goldschmidt_inverse(1,sum, epsilon=2**(-11), alpha=0.1/10)
    div_sum = goldschmidt_inverse(sum,iterations)
    return exp_approx * div_sum

def main():
    x_start = -24
    x_end= 24
    r = 12
    iterations = 22
    scale = 16
    delta = 4
    num_points = 100000
    
    print("==================================")
    np.random.seed(42)
    for i in range(1):
        x = np.random.uniform(-10, 10, 128)
        print(f"softmax误差对比分析:{i+1}轮")
        y_std = standard_softmax(x)
        y_thor = thor_softmax(x,(x_start,x_end),2,2)
        y_nexus = nexus_softmax(x-x_end,r,iterations,scale)
        y_my = my_softmax(x,delta,x_end,iterations)
        
        print("==================================")
        absolute_error = np.abs(y_std - y_nexus)
        relative_error = np.abs((y_std - y_nexus) / (y_std))
        print("nexus_softmax误差对比")
        print(f"sum:{np.sum(y_nexus)}")
        print(f"range:({np.min(y_nexus)},{np.max(y_nexus)})")
        print(f"最大误差: {np.max(absolute_error):.3e} ")
        print(f"平均绝对误差: {np.mean(absolute_error):.3e}")
        print(f"相对平均误差: {np.mean(relative_error) * 100:.2f}%")
        print(f"相对最大误差: {np.max(relative_error) * 100:.2f}%")
        print("==================================")
        
        absolute_error = np.abs(y_std - y_thor)
        relative_error = np.abs((y_std - y_thor) / (y_std))
        print("thor_softmax误差对比")
        print(f"sum:{np.sum(y_thor)}")
        print(f"range:({np.min(y_thor)},{np.max(y_thor)})")
        print(f"最大误差: {np.max(absolute_error):.3e} ")
        print(f"平均绝对误差: {np.mean(absolute_error):.3e}")
        print(f"相对平均误差: {np.mean(relative_error) * 100:.2f}%")
        print(f"相对最大误差: {np.max(relative_error) * 100:.2f}%")
        print("==================================")

        absolute_error = np.abs(y_std - y_my)
        relative_error = np.abs((y_std - y_my) / (y_std))
        print("my_softmax误差对比")
        print(f"sum:{np.sum(y_my)}")
        print(f"range:({np.min(y_my)},{np.max(y_my)})")
        print(f"最大误差: {np.max(absolute_error):.3e} ")
        print(f"平均绝对误差: {np.mean(absolute_error):.3e}")
        print(f"相对平均误差: {np.mean(relative_error) * 100:.2f}%")
        print(f"相对最大误差: {np.max(relative_error) * 100:.2f}%")
        print("==================================")
if __name__ == "__main__":
    main()