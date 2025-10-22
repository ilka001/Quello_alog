import csv
import statistics
import math

def is_numeric(value):
    """检查值是否为有效数字"""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def detect_outliers_iqr(data):
    """使用IQR方法检测异常值"""
    if len(data) < 4:
        return []
    
    data_sorted = sorted(data)
    q1 = statistics.quantiles(data_sorted, n=4)[0]
    q3 = statistics.quantiles(data_sorted, n=4)[2]
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = []
    for i, value in enumerate(data):
        if value < lower_bound or value > upper_bound:
            outliers.append(i)
    
    return outliers

def clean_data(input_file, output_file):
    """清洗数据"""
    print("开始数据清洗...")
    
    # 读取数据
    rows = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        headers = reader.fieldnames
        print(f"列名: {headers}")
        
        for row in reader:
            rows.append(row)
    
    print(f"原始数据行数: {len(rows)}")
    
    # 检查缺失值
    missing_stats = {}
    for header in headers:
        if header != 'emotion':  # 跳过非数值列
            missing_count = 0
            for row in rows:
                value = row[header].strip()
                if value == '' or value.lower() in ['nan', 'null', 'none', 'na']:
                    missing_count += 1
            missing_stats[header] = missing_count
    
    print("\n缺失值统计:")
    for col, count in missing_stats.items():
        if count > 0:
            print(f"{col}: {count} 个缺失值 ({count/len(rows)*100:.2f}%)")
        else:
            print(f"{col}: 无缺失值")
    
    # 检查异常值
    print("\n异常值检测 (使用IQR方法):")
    outlier_stats = {}
    
    for header in headers:
        if header != 'emotion':  # 跳过非数值列
            values = []
            valid_indices = []
            
            for i, row in enumerate(rows):
                value = row[header].strip()
                if is_numeric(value) and value != '':
                    values.append(float(value))
                    valid_indices.append(i)
            
            if len(values) > 0:
                outliers = detect_outliers_iqr(values)
                outlier_indices = [valid_indices[i] for i in outliers]
                outlier_stats[header] = outlier_indices
                
                if len(outliers) > 0:
                    print(f"{header}: {len(outliers)} 个异常值 ({len(outliers)/len(values)*100:.2f}%)")
                else:
                    print(f"{header}: 无异常值")
    
    # 数据清洗策略
    print("\n开始数据清洗...")
    
    # 1. 处理缺失值 - 用中位数填充
    for header in headers:
        if header != 'emotion' and missing_stats[header] > 0:
            values = []
            for row in rows:
                value = row[header].strip()
                if is_numeric(value) and value != '':
                    values.append(float(value))
            
            if values:
                median_value = statistics.median(values)
                print(f"用中位数 {median_value:.4f} 填充 {header} 的缺失值")
                
                for row in rows:
                    value = row[header].strip()
                    if value == '' or value.lower() in ['nan', 'null', 'none', 'na']:
                        row[header] = str(median_value)
    
    # 2. 处理异常值 - 用中位数替换
    for header in headers:
        if header != 'emotion' and header in outlier_stats:
            outlier_indices = outlier_stats[header]
            if outlier_indices:
                # 计算非异常值的中位数
                non_outlier_values = []
                for i, row in enumerate(rows):
                    if i not in outlier_indices:
                        value = row[header].strip()
                        if is_numeric(value):
                            non_outlier_values.append(float(value))
                
                if non_outlier_values:
                    median_value = statistics.median(non_outlier_values)
                    print(f"用中位数 {median_value:.4f} 替换 {header} 的 {len(outlier_indices)} 个异常值")
                    
                    for i in outlier_indices:
                        rows[i][header] = str(median_value)
    
    # 3. 最终验证
    print("\n最终数据验证:")
    final_missing = {}
    final_outliers = {}
    
    for header in headers:
        if header != 'emotion':
            values = []
            missing_count = 0
            
            for row in rows:
                value = row[header].strip()
                if value == '' or value.lower() in ['nan', 'null', 'none', 'na']:
                    missing_count += 1
                elif is_numeric(value):
                    values.append(float(value))
            
            final_missing[header] = missing_count
            
            if len(values) > 0:
                outliers = detect_outliers_iqr(values)
                final_outliers[header] = len(outliers)
    
    print("清洗后缺失值统计:")
    for col, count in final_missing.items():
        if count > 0:
            print(f"{col}: {count} 个缺失值")
        else:
            print(f"{col}: 无缺失值")
    
    print("\n清洗后异常值统计:")
    for col, count in final_outliers.items():
        if count > 0:
            print(f"{col}: {count} 个异常值")
        else:
            print(f"{col}: 无异常值")
    
    # 保存清洗后的数据
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n清洗后的数据已保存到: {output_file}")
    print(f"最终数据行数: {len(rows)}")
    
    # 显示清洗后的数据统计
    print("\n清洗后数据统计:")
    for header in headers:
        if header != 'emotion':
            values = []
            for row in rows:
                value = row[header].strip()
                if is_numeric(value):
                    values.append(float(value))
            
            if values:
                print(f"{header}:")
                print(f"  均值: {statistics.mean(values):.4f}")
                print(f"  中位数: {statistics.median(values):.4f}")
                print(f"  标准差: {statistics.stdev(values):.4f}")
                print(f"  最小值: {min(values):.4f}")
                print(f"  最大值: {max(values):.4f}")

if __name__ == "__main__":
    input_file = 'hrv_data_4_emotions_clean.csv'
    output_file = 'hrv_data_4_emotions_cleaned.csv'
    
    clean_data(input_file, output_file)
