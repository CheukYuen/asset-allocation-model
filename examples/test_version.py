# n8n 会调用 main 方法
# Args 在 n8n 中是内置类型，代表输入参数

import sys
import platform

def main(args: dict) -> dict:
    """
    n8n 主函数
    
    Args:
        args: 输入参数字典，包含 'input' 和 'input2'
    
    Returns:
        包含 'result' 的字典
    """
    # 获取输入参数
    input1 = args.get("input", 0)
    input2 = args.get("input2", 0)
    
    # 计算结果
    result = input1 + input2
    
    # 打印 Python 版本信息
    print(f"=== n8n Python 环境信息 ===")
    print(f"Python 版本: {sys.version}")
    print(f"Python 版本号: {platform.python_version()}")
    print(f"Python 实现: {platform.python_implementation()}")
    print(f"系统平台: {platform.platform()}")
    print(f"系统架构: {platform.machine()}")
    print(f"可执行文件路径: {sys.executable}")
    print(f"========================\n")
    
    # 打印调试信息
    print(f"=== 执行日志 ===")
    print(f"输入参数 args: {args}")
    print(f"input1: {input1}")
    print(f"input2: {input2}")
    print(f"计算结果: {result}")
    print(f"==================")
    
    # 返回结果
    return {
        "result": result
    }