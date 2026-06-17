import os

# 1. 读取原始文件
with open('preprocess_tetraview.py', 'r') as f:
    code = f.read()

# 2. 准备要替换的旧代码片段 (完全匹配你之前发给我的 tail 内容)
old_loop = '''    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        result = process_molecule(row, idx, tokenizer)
        
        if result is not None:
            output_path = os.path.join(OUTPUT_DIR, f'data_{idx}.pt')
            torch.save(result, output_path)
            successful += 1
        else:
            failed += 1'''

# 3. 准备新代码片段 (并行版)
new_loop = '''    # === 并行加速代码 ===
    n_cores = os.cpu_count()
    print(f"Using {n_cores} CPU cores...")
    
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_wrapper)(row, idx, tokenizer, OUTPUT_DIR) 
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parallel Processing")
    )
    
    successful = sum(results)
    failed = len(results) - successful'''

# 4. 准备辅助函数和 Import
header_imports = "from joblib import Parallel, delayed\nimport os\n"
wrapper_function = '''
def process_wrapper(row, idx, tokenizer, output_dir):
    try:
        result = process_molecule(row, idx, tokenizer)
        if result is not None:
            torch.save(result, os.path.join(output_dir, f'data_{idx}.pt'))
            return 1
        return 0
    except:
        return 0

'''

# 5. 开始组装
print("正在改造代码...")

# 添加头部引用
if "from joblib" not in code:
    code = header_imports + code

# 插入辅助函数 (放在 def main(): 之前)
code = code.replace("def main():", wrapper_function + "def main():")

# 替换循环
if old_loop in code:
    code = code.replace(old_loop, new_loop)
    print("成功替换循环逻辑！")
else:
    # 如果缩进有细微差别，尝试模糊替换
    print("注意：精确匹配失败，尝试智能替换...")
    start_marker = 'for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):'
    end_marker = 'failed += 1'
    if start_marker in code:
        parts = code.split(start_marker)
        pre_loop = parts[0]
        # 找到循环结束后的部分
        rest = parts[1]
        if end_marker in rest:
            post_loop = rest.split(end_marker, 1)[1]
            code = pre_loop + new_loop.strip() + post_loop
            print("智能替换成功！")
        else:
            print("错误：无法定位循环结束位置")
            exit(1)
    else:
        print("错误：无法找到旧循环代码，请检查源文件")
        exit(1)

# 6. 写入新文件
with open('preprocess_fast.py', 'w') as f:
    f.write(code)

print("新脚本 preprocess_fast.py 生成完毕！")
