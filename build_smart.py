import os

# 读取原文件
with open('preprocess_tetraview.py', 'r') as f:
    content = f.read()

# 1. 定义 Wrapper 函数 (用于断点续传)
wrapper_code = '''
def process_wrapper(row, idx, tokenizer, output_dir):
    # 检查文件是否存在，存在则跳过
    out_file = os.path.join(output_dir, f"data_{idx}.pt")
    if os.path.exists(out_file):
        return 1
    try:
        # 不存在则计算
        res = process_molecule(row, idx, tokenizer)
        if res is not None:
            torch.save(res, out_file)
            return 1
        return 0
    except:
        return 0
'''

# 2. 定义 32核 并行循环
loop_code = '''    # === Smart Mode (32 cores) ===
    n_jobs = 32
    print(f"Using {n_jobs} cores (Resume enabled)...")
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(process_wrapper)(row, idx, tokenizer, OUTPUT_DIR) 
        for idx, row in tqdm(df.iterrows(), total=len(df), mininterval=1.0)
    )
    successful = sum(results)
    failed = len(results) - successful
    # ============================'''

# 3. 开始组装
if "from joblib" not in content:
    content = "from joblib import Parallel, delayed\n" + content

content = content.replace("def main():", wrapper_code + "\n\ndef main():")

# 定位替换点
start_mark = 'print(f"\\nProcessing molecules...")'
end_mark = 'print("\\n" + "=" * 80)'

p1 = content.find(start_mark)
p2 = content.find(end_mark)

if p1 != -1 and p2 != -1:
    # 关键修正：在 content[p2:] 前面强制加 4 个空格，修复 IndentationError
    new_content = content[:p1+len(start_mark)] + "\n    successful=0\n    failed=0\n" + loop_code + "\n    " + content[p2:]
    
    with open('preprocess_smart.py', 'w') as f:
        f.write(new_content)
    print("修复成功！缩进已自动对齐。")
else:
    print("错误：无法定位代码位置，请检查原文件。")
