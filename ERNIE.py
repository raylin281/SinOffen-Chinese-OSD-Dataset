import pandas as pd
import requests
import json
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score
import time
import random

# 文心一言 API 密钥
API_KEY = "xxxx"
SECRET_KEY = "xxxx"

# 加载 CSV 文件
data = pd.read_csv('sinoffen.csv')  

# 确保数据中没有缺失值
data = data.dropna(subset=['content', 'label'])
data['label'] = data['label'].astype(int)  # 确保标签是整数

# 获取文心一言 access_token
def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

# Prompt 模板
#prompt_template = (
  #  "请判断以下句子是否包含攻击性内容。\n"
   # "'0'表示句子不包含攻击性内容，'1'表示句子包含攻击性内容。\n"
   # "请只回答0或1，不要添加任何其他内容。\n"
   # "句子：{text}\n结果："
#)

prompt_template = (
    "Please determine whether the following sentence contains offensive content.\n"
    "'0' indicates that the sentence is not offensive, and '1' indicates that it is offensive.\n"
    "Please provide your response (only '0' or '1') \n"
    "sentence：{text}\nresults："
)

# 生成预测的函数
def generate_prediction(text, max_retries=3, base_wait_time=5):
    prompt = prompt_template.format(text=text)
    
    for retry in range(max_retries):
        try:
            # 构建 API 请求
            url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-4.0-turbo-128k?access_token=" + get_access_token()
            
            payload = json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": "你是一个用于检测攻击性言论的助手，请只回答1或0"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "penalty_score": 1,
                "enable_system_memory": False,
                "disable_search": True,
                "enable_citation": False,
                "enable_trace": False
            }, ensure_ascii=False)
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
            response_json = json.loads(response.text)
            
            # 检查是否达到请求限制
            if 'error_code' in response_json and response_json['error_code'] == 17:
                wait_time = base_wait_time * (2 ** retry) + random.uniform(0, 1)
                print(f"API 请求限制，等待 {wait_time:.2f} 秒后重试 ({retry+1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            
            # 获取生成的文本
            if 'result' in response_json:
                generated_text = response_json['result']
                print(f"Generated Text: {generated_text}")  # 打印生成结果
                
                # 解析模型的输出
                if '0' in generated_text:
                    return 0
                elif '1' in generated_text:
                    return 1
                else:
                    return -1  # 解析失败
            else:
                print(f"API 返回异常: {response_json}")
                # 如果是其他类型的错误，也等待一段时间后重试
                if retry < max_retries - 1:
                    wait_time = base_wait_time * (2 ** retry) + random.uniform(0, 1)
                    print(f"等待 {wait_time:.2f} 秒后重试 ({retry+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    return -1
                
        except Exception as e:
            print(f"API 请求失败: {e}")
            if retry < max_retries - 1:
                wait_time = base_wait_time * (2 ** retry) + random.uniform(0, 1)
                print(f"等待 {wait_time:.2f} 秒后重试 ({retry+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                return -1  # 请求失败

    return -1  # 所有重试都失败

#  批量推理
print("开始预测并打印结果：")
results = []

# 如果存在保存的结果文件，则加载已有结果
import os
save_file = 'xxx.csv' #这里改成保存文件路径
start_index = 0

if os.path.exists(save_file):
    try:
        saved_data = pd.read_csv(save_file)
        # 找到已经处理过的最后一个样本索引
        start_index = len(saved_data)
        # 加载已有的结果
        results = saved_data['predicted_label'].tolist()
        print(f"已加载 {start_index} 个已处理样本的结果")
    except Exception as e:
        print(f"加载已有结果失败: {e}")
        start_index = 0
        results = []

# 从上次中断的地方继续处理
for index, row in data.iloc[start_index:].iterrows():
    text = row['content']
    true_label = row['label']
    predicted_label = generate_prediction(text)
    results.append(predicted_label)
    print(f"样本 {index + 1}:")
    print(f"文本: {text}")
    print(f"实际标签: {true_label}, 预测标签: {predicted_label}")
    print("-" * 50)
    
    # 添加短暂延迟以避免API速率限制
    time.sleep(2 + random.uniform(0, 1))
    
    # 每处理10个样本保存一次结果
    if (len(results) % 10 == 0) or (index == len(data) - 1):
        # 创建临时DataFrame保存当前结果
        temp_data = data.iloc[:len(results)].copy()
        temp_data['predicted_label'] = results
        temp_data.to_csv(save_file, index=False)
        print(f"已保存当前处理结果到 {save_file}")

# 将预测结果存储到 DataFrame
data['predicted_label'] = results

# 保存完整结果
data.to_csv(save_file, index=False)
print(f"已保存完整结果到 {save_file}")

# 过滤无法分类的样本
valid_data = data[data['predicted_label'] != -1]

# 评估模型性能
y_true = valid_data['label']
y_pred = valid_data['predicted_label']

# 计算 AUC
try:
    auc_score = roc_auc_score(y_true, y_pred)
    print(f"AUC: {auc_score:.4f}")
except ValueError:
    print("AUC 计算失败，可能是因为标签只有一个类别。")

# 打印分类报告
print("\n分类报告：")
print(classification_report(y_true, y_pred, target_names=["Non-Hate", "Hate"], digits=4))

# 计算每个类别的 Precision、Recall 和 F1
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1])

# 打印每个类别的具体指标
print("\n每个类别的指标：")
print(f"Non-Hate -> Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}")
print(f"Hate -> Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}")

# 计算宏平均
macro_precision = precision.mean()
macro_recall = recall.mean()
macro_f1 = f1.mean()

print("\n总体宏平均指标：")
print(f"Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}") 