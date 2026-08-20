import readline
import os
import re
import subprocess
import torch
import requests
from bs4 import BeautifulSoup
from transformers import AutoModelForCausalLM, AutoTokenizer

# 避免 Connection aborted

# 初始化模型
model_id = "Qwen/Qwen2.5-3B-Instruct"
cached_folder = "./local_models"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cached_folder)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cached_folder).to(device)
print("模型和分词器加载完成！\n")

def generate_response(messages, max_new_tokens=512):
    """通用的模型生成函数"""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        model_inputs.input_ids, 
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def execute_command_safely(command):
    """执行命令并捕获输出，加入长度截断防爆显存"""
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=15)
        output = result.stdout + result.stderr
        
        if not output.strip():
            return "[命令执行成功，但无输出]"
            
        if len(output) > 800:
            return output[:400] + "\n...[输出太长已截断]...\n" + output[-400:]
        return output
    except Exception as e:
        return f"[执行异常]: {str(e)}"

def search_internet_safely(query: str, max_results=5) -> str:
    """
    原生且安全的网页搜索模块 (0 API Key 依赖)。
    直接抓取 DuckDuckGo HTML 版前几个结果，等效于人工翻看前1-2页的精华。
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"
    }
    url = "https://html.duckduckgo.com/html/"
    try:
        response = requests.get(url, params={"q": query}, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        
        # 获取结果块，切片限制数量，避免抓取过多导致封禁或上下文溢出
        for i, block in enumerate(soup.find_all("div", class_="result__body")[:max_results]):
            title_tag = block.find("h2", class_="result__title")
            snippet_tag = block.find("a", class_="result__snippet")
            
            title = title_tag.text.strip() if title_tag else "无标题"
            snippet = snippet_tag.text.strip() if snippet_tag else "无摘要"
            results.append(f"[{i+1}] {title}\n摘要: {snippet}")
            
        if not results:
            return "[搜索完成，但未找到相关外部信息]"
        return "\n\n".join(results)
    except Exception as e:
        return f"[搜索异常]: 无法访问外部网络，错误信息 {e}"

# ==========================================
# 核心：交互式工作流 Agent
# ==========================================

# 强约束的系统提示词：教会模型使用两把“武器”
system_prompt = (
    "你是一个资深的 Linux 运维专家助手。请遵循以下严格规则排查问题：\n"
    "1. 你具备两种工具：执行服务器命令 和 搜索互联网资料。\n"
    "2. 如果你需要执行命令获取系统状态，请将命令放在 ```bash 和 ``` 之间。\n"
    "3. 如果遇到了你不懂的报错，或者需要查询官方文档，请将搜索关键词放在 ```search 和 ``` 之间。\n"
    "4. 每次回复中，最多只能使用一种工具，不要贪多。\n"
    "5. 如果排查结束，请直接用自然语言给出最终结论，不需要输出任何代码块。"
)

chat_history = [{"role": "system", "content": system_prompt}]

print("=== 交互式运维 Agent (联网增强版) 已启动 ===")
print("随时可以输入新的问题。Agent 会给出操作或搜索建议，等待您的审核。\n")

while True:
    user_input = input("\n👤 描述你的运维问题 (输入 'exit' 退出): ")
    if user_input.lower() in ['exit', 'quit']:
        break
    if not user_input.strip(): continue

    chat_history.append({"role": "user", "content": user_input})

    while True:
        print("\n🤖 [Agent 思考中...]")
        # 窗口滑动：保留系统提示词 + 最近的 8 条对话 (因加入搜索，上下文稍微放宽)
        if len(chat_history) > 9:
            chat_history = [chat_history[0]] + chat_history[-8:]
            
        response = generate_response(chat_history)
        print("--------------------------------------------------")
        print(response)
        print("--------------------------------------------------")
        
        chat_history.append({"role": "assistant", "content": response})

        # 分支 1：匹配 Bash 执行请求
        match_bash = re.search(r'```bash\n(.*?)\n```', response, re.DOTALL)
        # 分支 2：匹配 联网搜索 请求
        match_search = re.search(r'```search\n(.*?)\n```', response, re.DOTALL)
        
        if match_bash:
            suggested_cmd = match_bash.group(1).strip()
            print(f"\n Agent 提议执行下一步命令: \033[93m{suggested_cmd}\033[0m")
            action = input("您是否允许执行？ [y]执行 / [n]拒绝 / [m]手动输入: ").strip().lower()
            
            if action == 'y':
                print("=== 正在执行...")
                cmd_output = execute_command_safely(suggested_cmd)
                print(f"=== 执行结果:\n{cmd_output}")
                feedback_msg = f"这是命令的执行结果:\n```\n{cmd_output}\n```\n请根据结果决定下一步。"
                chat_history.append({"role": "user", "content": feedback_msg})
                continue 
            elif action == 'm':
                manual_output = input("请粘贴执行结果: ")
                feedback_msg = f"手动执行命令返回:\n```\n{manual_output}\n```"
                chat_history.append({"role": "user", "content": feedback_msg})
                continue
            else:
                print("=== 已拒绝执行。")
                chat_history.append({"role": "user", "content": "我已经拒绝执行该命令，请换个思路。"})
                continue
                
        elif match_search:
            search_query = match_search.group(1).strip()
            print(f"\n Agent 请求联网搜索: \033[96m{search_query}\033[0m")
            
            # 对于搜索操作，一般不需要高危审核，直接自动放行
            print("=== 正在检索互联网...")
            search_output = search_internet_safely(search_query, max_results=5)
            
            feedback_msg = f"这是联网搜索返回的参考资料:\n```\n{search_output}\n```\n请结合这些资料继续解答我的问题。"
            chat_history.append({"role": "user", "content": feedback_msg})
            # 获取资料后自动继续让模型思考
            continue
            
        else:
            # 如果既没有 ```bash 也没有 ```search，说明思考闭环结束
            break