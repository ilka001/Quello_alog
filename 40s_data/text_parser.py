#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本解析脚本
将包含时间范围和情绪的文本解析为简化的情绪标签格式
支持多种输入格式的解析和转换
"""

import re
from typing import List

# --------------------------------------------------------------------------------------------
# 文本解析函数
# --------------------------------------------------------------------------------------------

def parse_text_to_labels(text_content: str) -> List[str]:
    """将包含时间范围和情绪的文本解析为简化的情绪标签"""
    lines = text_content.strip().split('\n')
    results = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 跳过不包含时间范围的行（标签行）
        if not re.search(r'\d+-\d+', line):
            continue
        
        # 主要模式：匹配 时间范围（情绪）数字 格式
        pattern = r'\d+-\d+\s*（([^）]+)）\s*(\d*\.?\d*)'
        match = re.search(pattern, line)
        
        if match:
            emotion = match.group(1).strip()
            number = match.group(2).strip()
            
            if number:
                result = f"{emotion} {number}"
            else:
                result = emotion
            
            results.append(result)
        else:
            # 备用模式：只匹配情绪，没有数字
            pattern2 = r'\d+-\d+\s*（([^）]+)）'
            match2 = re.search(pattern2, line)
            
            if match2:
                emotion = match2.group(1).strip()
                results.append(emotion)
    
    return results

def get_text_input() -> str:
    """从用户输入获取文本内容"""
    print("请输入需要解析的文本内容:")
    print("格式示例:")
    print("愉悦 570000-750000（愉悦）5 1140000-1800000（愉悦）7")
    print("平静悲伤 240000-660000（平静） 780000-1200000（平静）")
    print("焦虑 210000-1620000（焦虑）7 1770000-2160000（焦虑）8")
    print("...")
    print("输入完成后按Ctrl+Z然后回车结束")
    print("注意：结果将直接显示在控制台，不会保存到文件")
    print()
    
    lines = []
    try:
        while True:
            line = input().strip()
            if line:  # 忽略空行
                lines.append(line)
    except EOFError:
        pass
    
    return '\n'.join(lines)

# --------------------------------------------------------------------------------------------
# 主处理函数
# --------------------------------------------------------------------------------------------

def main():
    print("=== 文本解析脚本 ===")
    print("将包含时间范围和情绪的文本解析为简化的情绪标签")
    print()
    
    # 获取文本输入
    text_content = get_text_input()
    
    if not text_content.strip():
        print("❌ 未输入任何内容")
        return
    
    print(f"\n📝 输入的原始文本:")
    print("-" * 80)
    print(text_content)
    print("-" * 80)
    print()
    
    # 解析文本
    parsed_results = parse_text_to_labels(text_content)
    
    if not parsed_results:
        print("❌ 未解析到任何有效数据")
        return
    
    print("=== 解析结果 ===")
    for result in parsed_results:
        print(result)
    
    print(f"\n📊 处理统计:")
    print(f"   输入行数: {len(text_content.strip().split(chr(10)))}")
    print(f"   解析结果: {len(parsed_results)}")

# --------------------------------------------------------------------------------------------
# 使用示例和说明
# --------------------------------------------------------------------------------------------

def show_usage_example():
    """显示使用示例"""
    print("=== 使用示例 ===")
    print("输入格式:")
    print("  愉悦 570000-750000（愉悦）5 1140000-1800000（愉悦）7")
    print("  平静悲伤 240000-660000（平静） 780000-1200000（平静）")
    print("  焦虑 210000-1620000（焦虑）7 1770000-2160000（焦虑）8")
    print()
    print("输出格式:")
    print("  愉悦 5")
    print("  愉悦 7")
    print("  平静")
    print("  平静")
    print("  焦虑 7")
    print("  焦虑 8")
    print()
    print("支持的格式:")
    print("  - 时间范围（情绪）数字")
    print("  - 时间范围（情绪）")
    print("  - 支持整数和小数")
    print("  - 自动提取情绪和数字")

def test_parser():
    """测试解析功能"""
    print("=== 测试解析功能 ===")
    
    test_cases = [
        "愉悦 570000-750000（愉悦）5 1140000-1800000（愉悦）7",
        "平静悲伤 240000-660000（平静） 780000-1200000（平静）",
        "焦虑 210000-1620000（焦虑）7 1770000-2160000（焦虑）8",
        "810000-1620000（愉悦）7 3630000-4260000 （愉悦）4",
        "4620000-5580000 （平静） 7980000-8400000（平静）"
    ]
    
    print("测试用例:")
    for i, test_case in enumerate(test_cases, 1):
        print(f"  {i}. {test_case}")
    
    print("\n解析结果:")
    for i, test_case in enumerate(test_cases, 1):
        result = parse_text_to_labels(test_case)
        print(f"  {i}. {result}")
    
    print()

def debug_parsing():
    """调试解析过程"""
    print("=== 调试解析过程 ===")
    
    test_text = """愉悦 570000-750000（愉悦）5 1140000-1800000（愉悦）7
平静悲伤 240000-660000（平静） 780000-1200000（平静）
焦虑 210000-1620000（焦虑）7 1770000-2160000（焦虑）8"""
    
    print("测试文本:")
    print(test_text)
    print()
    
    lines = test_text.strip().split('\n')
    print("逐行解析:")
    for i, line in enumerate(lines, 1):
        print(f"  行 {i}: {line}")
        
        # 检查是否包含时间范围
        has_time_range = bool(re.search(r'\d+-\d+', line))
        print(f"    包含时间范围: {has_time_range}")
        
        if has_time_range:
            # 主要模式匹配
            pattern = r'\d+-\d+\s*（([^）]+)）\s*(\d*\.?\d*)'
            matches = re.findall(pattern, line)
            print(f"    匹配结果: {matches}")
            
            for match in matches:
                emotion = match[0].strip()
                number = match[1].strip()
                if number:
                    result = f"{emotion} {number}"
                else:
                    result = emotion
                print(f"      解析: {result}")
        print()

if __name__ == "__main__":
    # 显示使用说明
    show_usage_example()
    print("\n" + "="*60 + "\n")
    
    # 运行测试
    test_parser()
    
    # 运行调试
    debug_parsing()
    
    # 运行主程序
    main()
