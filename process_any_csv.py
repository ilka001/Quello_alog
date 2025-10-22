import csv
import sys
import os

def process_emotion_label(emotion_text):
    """
    处理情绪标签：
    1. 去掉人名
    2. 去掉分数
    3. 仅保留二字情绪标签
    4. 将大悲等改为悲伤
    """
    if not emotion_text:
        return ""
    
    # 去掉人名（第一个词）
    parts = emotion_text.split()
    if len(parts) <= 1:
        return ""
    
    # 去掉分数（最后一个数字）
    emotion_part = parts[1]
    
    # 处理特殊情况：大悲 -> 悲伤
    if emotion_part == "大悲":
        return "悲伤"
    elif emotion_part == "小悲":
        return "悲伤"
    elif emotion_part == "悲":
        return "悲伤"
    elif emotion_part == "小开心":
        return "开心"
    elif emotion_part == "开心":
        return "开心"
    elif emotion_part == "愉悦":
        return "开心"
    elif emotion_part == "平静":
        return "平静"
    elif emotion_part == "焦虑":
        return "焦虑"
    elif emotion_part == "紧张":
        return "紧张"
    else:
        return emotion_part

def process_csv_file(input_file, output_file):
    """处理CSV文件"""
    print(f"正在处理文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # 读取第一行数据来检查结构
        first_row = next(reader)
        print(f"第一行数据: {first_row}")
        
        # 根据数据特征判断列结构
        if len(first_row) > 7:  # 如果列数超过7，说明第一列是文件名
            new_header = ['RMSSD', 'pNN58', 'SDNN', 'SD1', 'SD2', 'SD1_SD2', 'SampEn', 'emotion']
            print("检测到第一列是文件名，将去掉第一列")
            remove_first_column = True
        else:
            new_header = ['RMSSD', 'pNN58', 'SDNN', 'SD1', 'SD2', 'SD1_SD2', 'SampEn', 'emotion']
            print("未检测到文件名列")
            remove_first_column = False
        
        writer.writerow(new_header)
        print(f"新列名: {new_header}")
        
        processed_count = 0
        emotion_counts = {}
        
        # 处理所有行数据
        all_rows = [first_row] + list(reader)
        
        for row in all_rows:
            if len(row) >= 2:  # 确保有足够的列
                # 检查是否需要去掉第一列
                if remove_first_column and len(row) > 7:
                    new_row = row[1:-1]  # 去掉第一列和最后一列
                else:
                    new_row = row[:-1]  # 只去掉最后一列
                
                processed_emotion = process_emotion_label(row[-1])  # 处理最后一列
                new_row.append(processed_emotion)  # 添加处理后的情绪标签
                
                writer.writerow(new_row)
                processed_count += 1
                
                # 统计情绪分布
                if processed_emotion in emotion_counts:
                    emotion_counts[processed_emotion] += 1
                else:
                    emotion_counts[processed_emotion] = 1
        
        print(f"\n处理完成！")
        print(f"总共处理了 {processed_count} 行数据")
        print(f"处理后的文件保存为: {output_file}")
        print(f"\n情绪分布:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"  {emotion}: {count}条")

def main():
    """主函数"""
    print("通用CSV文件处理脚本")
    print("="*50)
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if len(sys.argv) > 2:
            output_file = sys.argv[2]
        else:
            # 自动生成输出文件名
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_processed.csv"
    else:
        # 默认处理svm目录下的文件
        input_file = 'svm/2.csv'
        output_file = 'svm/2_processed.csv'
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在")
        return
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print()
    
    process_csv_file(input_file, output_file)
    
    print("\n处理完成！")

if __name__ == "__main__":
    main()
