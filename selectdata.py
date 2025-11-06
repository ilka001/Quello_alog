# 从hrv_data_10s_fortrain.csv 文档中筛选以下几个人的数据（每个人数据条数大于1000，同时有着4种情绪标签）
# 张卜依, 朱必艳, 朱燕, 朱颖, 李佳宜, 梁怡, 王滋雨, 赵紫涵, 邓雯, 陈盈盈, 韩乐蓉
# 把原先的格式：73.674163	35.294118	49.191403	51.915589	46.307233	1.121112	2.351375	赵紫涵	小悲伤	5 
# 转换新的格式：73.674163	35.294118	49.191403	51.915589	46.307233	1.121112	赵紫涵	悲伤
# 然后保存为seleted10.csv的文件

# 映射规则：
# 愉悦：'愉悦' '小愉悦'
# 悲伤：'小悲伤' '悲伤' 
# 焦虑：'焦虑'  '烦躁' '紧张'
# 平静：'平静' '几乎不焦虑' '放松'




import pandas as pd

# --- 配置信息 ---
INPUT_FILE = 'hrv_data_10s_fortrain.csv'
OUTPUT_FILE = 'seleted11.csv'
NAMES_TO_KEEP = [
    '张卜依', '朱必艳', '朱燕', '朱颖', '李佳宜', '梁怡', 
    '王滋雨', '赵紫涵', '邓雯', '陈盈盈', '韩乐蓉'
]
EMOTION_MAP = {
    '愉悦': '愉悦', '小愉悦': '愉悦',
    '小悲伤': '悲伤', '悲伤': '悲伤',
    '焦虑': '焦虑', '烦躁': '焦虑', '紧张': '焦虑',
    '平静': '平静', '几乎不焦虑': '平静', '放松': '平静'
}

def process_hrv_data(input_path, output_path, names, emotion_mapping):
    """
    加载、处理并保存HRV数据。

    :param input_path: 输入CSV文件的路径。
    :param output_path: 输出CSV文件的路径。
    :param names: 需要保留的受试者姓名列表。
    :param emotion_mapping: 情绪标签的映射规则。
    """
    # 1. 加载并解析数据
    df = pd.read_csv(input_path)
    df[['subject', 'emotion_new', 'score']] = df['emotion'].str.extract(r'(\S+)\s(\S+)\s?(\d*)?')
    df = df.drop(columns=['emotion', 'score','SampEn']).rename(columns={'emotion_new': 'emotion'})

    # 2. 筛选指定人员并应用情绪映射
    df_selected = df[df['subject'].isin(names)].copy()
    df_selected['emotion'] = df_selected['emotion'].replace(emotion_mapping)

    # 3. 保存处理后的数据
    df_selected.to_csv(output_path, index=False)
    print(f"文件 '{output_path}' 已保存完成。")

if __name__ == "__main__":
    process_hrv_data(INPUT_FILE, OUTPUT_FILE, NAMES_TO_KEEP, EMOTION_MAP)
