#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多线程数据划分脚本
根据文本输入将CSV文件按时间范围进行划分
支持多线程并行处理
"""

import os
import re
import glob
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import time

# --------------------------------------------------------------------------------------------
# 配置区域 - 用户需要修改的部分
# --------------------------------------------------------------------------------------------

# CSV文件路径列表（按顺序对应标签）
CSV_PATHS = [

    r"C:\Users\QAQ\Desktop\emotion\spe d\HOR\K.csv",
    r"C:\Users\QAQ\Desktop\emotion\spe d\HOR\F.csv",
    # r"", 
    # r"",
    # r"",
    # r"",     
    # r"",
    # r"",
    # r"", 
    # r"",
    # r"",
    # r"", 
    # r"",
    # r"",
    # r"",     
    # r"",
    # r"",
    # r"", 
    # r"",
    # r"",
    # r"", 
    # r"",
    # r"",
    # r"",     
    # r"",
    # r"",
    # r"", 
    # r"",
    # r"",
    # r"", 
    # r"",
    # r"",
    # r"",     
    # r"",
    # r"",
    # r"", 
    # r"",
    # r"",
    # r"", 
    # r"",
    # r"",
    # r"",     
    # r"",
    # r"",
    # r"", 
    # r"",
    # r"",
    # r"", 
    # r"",
    # r"",
    # r"", 
]

# 输出目录
OUTPUT_DIR = r"C:\Users\UiNCeY\Desktop\emotion\processed_data"

# 最大线程数
MAX_WORKERS = 14

# --------------------------------------------------------------------------------------------
# 文本解析和格式化函数
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

# --------------------------------------------------------------------------------------------
# 文本解析函数
# --------------------------------------------------------------------------------------------

def parse_text_file(text_content: str, csv_paths: List[str]) -> List[Tuple[str, int, str]]:
    """解析文本文件，返回(时间范围字符串, CSV索引, 输出文件名)的列表"""
    lines = text_content.strip().split('\n')
    tasks = []
    
    current_csv_index = -1
    # 为每个CSV索引维护一个计数器
    csv_counters = {i: 0 for i in range(len(csv_paths))}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 检查是否是标签行（不包含时间范围的行）
        if not re.search(r'\d+-\d+', line):
            print(f"📋 发现标签: {line} -> 切换到CSV索引 {current_csv_index}")
            current_csv_index += 1
            if current_csv_index >= len(csv_paths):
                print(f"⚠️ 警告: CSV索引 {current_csv_index} 超出范围，使用最后一个CSV文件")
                current_csv_index = len(csv_paths) - 1
            continue
        
        # 解析时间范围
        time_range_match = re.search(r'(\d+)-(\d+)', line)
        if time_range_match:
            start_time = int(time_range_match.group(1))
            end_time = int(time_range_match.group(2))
            
            # 验证时间范围
            if start_time >= end_time:
                print(f"⚠️ 警告: 无效时间范围 {start_time}-{end_time}，跳过")
                continue
            
            # 为当前CSV索引生成文件名
            csv_counters[current_csv_index] += 1
            output_filename = f"{current_csv_index + 1}{csv_counters[current_csv_index]}.csv"
            
            tasks.append((line, current_csv_index, output_filename))
            print(f"📝 任务: {line} -> CSV索引 {current_csv_index} -> {output_filename}")
        else:
            print(f"⚠️ 警告: 无法解析时间范围: {line}")
    
    return tasks

def parse_text_input() -> str:
    """从用户输入获取文本内容"""
    print("请输入文本内容（包含标签和时间范围）:")
    print("格式示例:")
    print("开")
    print("750000-1290000 （平静）")
    print("1630000-1980000 （平静）")
    print("2220000-2430000（紧张）6")
    print("平悲")
    print("60000-540000（平静）")
    print("630000-1050000（平静）")
    print("焦虑")
    print("3210000-4230000（焦虑）8")
    print("...")
    print("输入完成后按Ctrl+Z然后回车结束")
    print("注意：处理完成后会显示格式化的情绪标签")
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
# 数据处理函数
# --------------------------------------------------------------------------------------------

def split_csv_by_time_range(csv_path: str, start_time: int, end_time: int, 
                           output_filename: str, thread_id: int) -> bool:
    """根据时间范围划分CSV文件"""
    try:
        print(f"🔄 线程 {thread_id}: 开始处理 {os.path.basename(csv_path)} -> {output_filename}")
        
        # 读取CSV文件
        df = pd.read_csv(csv_path, header=None)
        
        if df.shape[1] < 2:
            print(f"❌ 线程 {thread_id}: 文件格式错误 {os.path.basename(csv_path)} (需要至少2列数据)")
            return False
        
        # 设置列名
        df = df.iloc[:, :2].copy()
        df.columns = ['时间', '数值']
        
        # 根据时间范围筛选数据
        mask = (df['时间'] >= start_time) & (df['时间'] <= end_time)
        filtered_df = df[mask].copy()
        
        if len(filtered_df) == 0:
            print(f"⚠️ 线程 {thread_id}: 时间范围 {start_time}-{end_time} 内无数据")
            return False
        
        # 保存划分后的数据
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        filtered_df.to_csv(output_path, index=False, header=False)
        
        print(f"✅ 线程 {thread_id}: 完成 {os.path.basename(csv_path)} -> {output_filename} ({len(filtered_df)} 行数据)")
        return True
        
    except Exception as e:
        print(f"❌ 线程 {thread_id}: 处理失败 {os.path.basename(csv_path)} -> {e}")
        return False

def process_single_task(task: Tuple[str, int, str], thread_id: int) -> bool:
    """处理单个划分任务"""
    time_range_str, csv_index, output_filename = task
    
    # 解析时间范围
    time_range_match = re.search(r'(\d+)-(\d+)', time_range_str)
    if not time_range_match:
        print(f"❌ 线程 {thread_id}: 无法解析时间范围 {time_range_str}")
        return False
    
    start_time = int(time_range_match.group(1))
    end_time = int(time_range_match.group(2))
    
    # 检查CSV索引是否有效
    if csv_index >= len(CSV_PATHS):
        print(f"❌ 线程 {thread_id}: CSV索引 {csv_index} 超出范围")
        return False
    
    csv_path = CSV_PATHS[csv_index]
    
    # 执行划分（文件名已经在解析阶段确定）
    return split_csv_by_time_range(csv_path, start_time, end_time, output_filename, thread_id)

# --------------------------------------------------------------------------------------------
# 多线程处理类
# --------------------------------------------------------------------------------------------

class MultiThreadDataSplitter:
    def __init__(self, csv_paths: List[str], output_dir: str, max_workers: int = 8):
        self.csv_paths = csv_paths
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.results = []
        self.processing_status = {}
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 创建输出目录: {output_dir}")
    
    def process_all_tasks(self, tasks: List[Tuple[str, int, str]], text_content: str = "") -> None:
        """多线程处理所有划分任务"""
        if not tasks:
            print("❌ 没有任务需要处理")
            return
        
        print(f"📊 开始多线程处理 {len(tasks)} 个任务")
        print(f"🔧 使用 {self.max_workers} 个线程")
        print(f"📁 输出目录: {self.output_dir}")
        print()
        
        start_time = time.time()
        success_count = 0
        
        # 初始化处理状态
        for i, (task, _, _) in enumerate(tasks):
            self.processing_status[i] = "等待中"
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {}
            for i, task in enumerate(tasks):
                future = executor.submit(process_single_task, task, i % self.max_workers)
                future_to_task[future] = (i, task)
            
            # 处理完成的任务
            completed_count = 0
            for future in as_completed(future_to_task):
                task_index, task = future_to_task[future]
                
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                        self.processing_status[task_index] = "✅ 成功"
                    else:
                        self.processing_status[task_index] = "❌ 失败"
                except Exception as e:
                    print(f"❌ 任务执行异常: {task} - {e}")
                    self.processing_status[task_index] = "❌ 异常"
                
                completed_count += 1
                
                # 显示进度
                progress = (completed_count / len(tasks)) * 100
                elapsed_time = time.time() - start_time
                tasks_per_second = completed_count / elapsed_time if elapsed_time > 0 else 0
                
                print(f"\r🔄 处理进度: {completed_count}/{len(tasks)} ({progress:.1f}%) - 成功: {success_count}, 失败: {completed_count - success_count} - 速度: {tasks_per_second:.1f} 任务/秒", end="", flush=True)
        
        end_time = time.time()
        
        print(f"\n=== 处理完成 ===")
        print(f"总任务数: {len(tasks)}")
        print(f"成功处理: {success_count}")
        print(f"失败任务: {len(tasks) - success_count}")
        print(f"处理时间: {end_time - start_time:.2f} 秒")
        
        # 显示详细处理结果
        self._show_detailed_results()
        
        # 显示格式化的情绪标签
        self._show_formatted_labels(text_content)
    
    def _show_detailed_results(self):
        """显示详细处理结果"""
        print(f"\n📋 详细处理结果:")
        print("-" * 80)
        
        # 按状态分组显示
        status_groups = {}
        for task_index, status in self.processing_status.items():
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append(task_index)
        
        for status, task_indices in status_groups.items():
            print(f"{status}: {len(task_indices)} 个任务")
            for task_index in sorted(task_indices):
                print(f"  - 任务 {task_index + 1}")
            print()
    
    def _show_formatted_labels(self, text_content: str):
        """显示格式化的情绪标签"""
        if not text_content.strip():
            return
        
        print(f"\n🏷️ 格式化情绪标签:")
        print("-" * 40)
        
        # 解析情绪标签
        formatted_labels = parse_text_to_labels(text_content)
        
        if formatted_labels:
            for label in formatted_labels:
                print(label)
        else:
            print("未解析到任何情绪标签")
        
        print("-" * 40)

# --------------------------------------------------------------------------------------------
# 主处理函数
# --------------------------------------------------------------------------------------------

def main():
    print("=== 多线程数据划分脚本 ===")
    print(f"CSV文件数量: {len(CSV_PATHS)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"最大线程数: {MAX_WORKERS}")
    print()
    
    # 显示CSV文件列表
    print("📁 CSV文件列表:")
    for i, csv_path in enumerate(CSV_PATHS, 1):
        if os.path.exists(csv_path):
            print(f"   {i}. {os.path.basename(csv_path)} ✅")
        else:
            print(f"   {i}. {os.path.basename(csv_path)} ❌ (文件不存在)")
    print()
    
    # 检查CSV文件是否存在
    missing_files = [csv_path for csv_path in CSV_PATHS if not os.path.exists(csv_path)]
    if missing_files:
        print("❌ 以下CSV文件不存在:")
        for csv_path in missing_files:
            print(f"   - {csv_path}")
        print("请检查文件路径是否正确")
        return
    
    # 获取文本输入
    text_content = parse_text_input()
    
    if not text_content.strip():
        print("❌ 未输入任何内容")
        return
    
    print(f"\n📝 输入的文本内容:")
    print("-" * 40)
    print(text_content)
    print("-" * 40)
    print()
    
    # 解析文本
    tasks = parse_text_file(text_content, CSV_PATHS)
    
    if not tasks:
        print("❌ 未解析到任何有效任务")
        return
    
    print(f"📋 解析到 {len(tasks)} 个任务:")
    for i, (time_range, csv_index, output_filename) in enumerate(tasks, 1):
        csv_name = os.path.basename(CSV_PATHS[csv_index])
        print(f"   {i}. {time_range} -> {csv_name} -> {output_filename}")
    print()
    
    # 创建处理器并开始处理
    splitter = MultiThreadDataSplitter(CSV_PATHS, OUTPUT_DIR, MAX_WORKERS)
    splitter.process_all_tasks(tasks, text_content)
    
    print(f"\n🎉 所有处理完成！")
    print(f"📁 输出文件保存在: {OUTPUT_DIR}")

# --------------------------------------------------------------------------------------------
# 使用示例和说明
# --------------------------------------------------------------------------------------------

def show_usage_example():
    """显示使用示例"""
    print("=== 使用示例 ===")
    print("1. 确保CSV文件路径正确")
    print("2. 运行脚本后按提示输入文本:")
    print("   开")
    print("   750000-1290000 （平静）")
    print("   1630000-1980000 （平静）")
    print("   2220000-2430000（紧张）6")
    print("   平悲")
    print("   60000-540000（平静）")
    print("   630000-1050000（平静）")
    print("   焦虑")
    print("   3210000-4230000（焦虑）8")
    print("3. 脚本会自动:")
    print("   - 解析文本中的标签和时间范围")
    print("   - 根据标签顺序分配CSV文件")
    print("   - 多线程并行处理所有划分任务")
    print("   - 保存结果到processed_data目录")
    print("4. 输出文件命名: 11.csv, 12.csv, 21.csv, 22.csv...")
    print("   (第一个数字表示CSV索引，第二个数字表示该CSV的第几个划分)")
    print("   开标签 -> 11.csv, 12.csv, 13.csv...")
    print("   平悲标签 -> 21.csv, 22.csv, 23.csv...")
    print("   焦虑标签 -> 31.csv, 32.csv, 33.csv...")

if __name__ == "__main__":
    # 显示使用说明
    show_usage_example()
    print("\n" + "="*60 + "\n")
    
    # 运行主程序
    main()
