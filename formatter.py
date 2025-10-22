#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式化脚本
将时间描述文本转换为毫秒时间范围格式
支持多种输入格式的解析和转换
"""

import re
from typing import List

# --------------------------------------------------------------------------------------------
# 时间转换函数
# --------------------------------------------------------------------------------------------

def parse_time_to_milliseconds(time_str: str) -> int:
    """将时间字符串转换为毫秒"""
    try:
        # 支持整数和小数
        time_float = float(time_str)
        return int(time_float * 60 * 1000)  # 分钟转毫秒
    except ValueError:
        return 0

def parse_duration_to_milliseconds(duration_str: str) -> int:
    """将持续时间字符串转换为毫秒"""
    try:
        # 支持整数和小数
        duration_float = float(duration_str)
        return int(duration_float * 60 * 1000)  # 分钟转毫秒
    except ValueError:
        return 0

# --------------------------------------------------------------------------------------------
# 文本解析和格式化函数
# --------------------------------------------------------------------------------------------

def format_data(text_content: str) -> List[str]:
    """将时间描述文本转换为毫秒时间范围格式"""
    lines = text_content.strip().split('\n')
    results = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 尝试格式1：有"开始"的格式
        pattern1 = r'([^0-9]+)(\d+)分[，：]\s*第(\d+(?:\.\d+)?)分钟开始往后(\d+(?:\.\d+)?)分钟'
        match1 = re.search(pattern1, line)
        
        if match1:
            emotion = match1.group(1).strip()
            score = match1.group(2)
            start_minutes = match1.group(3)
            duration_minutes = match1.group(4)
            
            start_ms = parse_time_to_milliseconds(start_minutes)
            duration_ms = parse_duration_to_milliseconds(duration_minutes)
            end_ms = start_ms + duration_ms
            
            result_line = f"{start_ms}-{end_ms} （{emotion}） {score}"
            results.append(result_line)
        else:
            # 尝试格式2：没有"开始"的格式
            pattern2 = r'([^0-9]+)(\d+)分[，：]\s*第(\d+(?:\.\d+)?)分钟往后(\d+(?:\.\d+)?)分钟'
            match2 = re.search(pattern2, line)
            
            if match2:
                emotion = match2.group(1).strip()
                score = match2.group(2)
                start_minutes = match2.group(3)
                duration_minutes = match2.group(4)
                
                start_ms = parse_time_to_milliseconds(start_minutes)
                duration_ms = parse_duration_to_milliseconds(duration_minutes)
                end_ms = start_ms + duration_ms
                
                result_line = f"{start_ms}-{end_ms} （{emotion}） {score}"
                results.append(result_line)
            else:
                # 如果都不匹配，保留原行
                results.append(line)
    
    return results

def get_text_input() -> str:
    """从用户输入获取文本内容"""
    print("请输入需要格式化的文本内容:")
    print("格式示例:")
    print("愉悦5分，第29.5分钟开始往后3.5分钟")
    print("愉悦7分，第50分钟开始往后8.5分钟")
    print("平静5分，第5分钟开始往后8分钟")
    print("焦虑5分：第18分钟开始往后4.5分钟")
    print("...")
    print("输入完成后按Ctrl+Z然后回车结束")
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
    print("=== 数据格式化脚本 ===")
    print("将时间描述文本转换为毫秒时间范围格式")
    print()
    
    # 获取文本输入
    text_content = get_text_input()
    
    if not text_content.strip():
        print("❌ 未输入任何内容")
        return
    
    print(f"\n📝 输入的原始文本:")
    print("-" * 60)
    print(text_content)
    print("-" * 60)
    print()
    
    # 格式化数据
    formatted_results = format_data(text_content)
    
    if not formatted_results:
        print("❌ 未解析到任何有效数据")
        return
    
    print("=== 格式化结果 ===")
    for i, result in enumerate(formatted_results, 1):
        print(f"{i}. {result}")
    
    print(f"\n📊 处理统计:")
    print(f"   输入行数: {len(text_content.strip().split('\\n'))}")
    print(f"   输出行数: {len(formatted_results)}")
    
    # 询问是否保存到文件
    save_to_file = input("\n是否保存结果到文件？(y/n，默认n): ").strip().lower()
    if save_to_file in ['y', 'yes', '是']:
        output_file = "formatted_data.txt"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in formatted_results:
                    f.write(result + '\\n')
            print(f"✅ 结果已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")

# --------------------------------------------------------------------------------------------
# 使用示例和说明
# --------------------------------------------------------------------------------------------

def show_usage_example():
    """显示使用示例"""
    print("=== 使用示例 ===")
    print("输入格式:")
    print("  愉悦5分，第29.5分钟开始往后3.5分钟")
    print("  愉悦7分，第50分钟开始往后8.5分钟")
    print("  平静5分，第5分钟开始往后8分钟")
    print("  焦虑5分：第18分钟开始往后4.5分钟")
    print()
    print("输出格式:")
    print("  1770000-1980000 （愉悦） 5")
    print("  3000000-3510000 （愉悦） 7")
    print("  300000-780000 （平静） 5")
    print("  1080000-1350000 （焦虑） 5")
    print()
    print("支持的格式:")
    print("  - 支持逗号和冒号分隔符")
    print("  - 支持有'开始'和无'开始'的格式")
    print("  - 支持整数和小数分钟")
    print("  - 自动转换为毫秒时间范围")

def test_formatter():
    """测试格式化功能"""
    print("=== 测试格式化功能 ===")
    
    test_cases = [
        "愉悦5分，第29.5分钟开始往后3.5分钟",
        "愉悦7分，第50分钟开始往后8.5分钟",
        "平静5分，第5分钟开始往后8分钟",
        "焦虑5分：第18分钟开始往后4.5分钟",
        "悲伤3分，第25分钟往后2分钟"
    ]
    
    print("测试用例:")
    for i, test_case in enumerate(test_cases, 1):
        print(f"  {i}. {test_case}")
    
    print("\n格式化结果:")
    for i, test_case in enumerate(test_cases, 1):
        result = format_data(test_case)
        print(f"  {i}. {result[0] if result else '解析失败'}")
    
    print()

if __name__ == "__main__":
    # 显示使用说明
    show_usage_example()
    print("\n" + "="*60 + "\n")
    
    # 运行测试
    test_formatter()
    
    # 运行主程序
    main()
