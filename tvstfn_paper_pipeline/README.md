# TV-STFN Paper Pipeline

本目录将论文后半部分工作包（WP0-WP5）拆分为独立脚本，并统一输出到：

- tvstfn_paper_pipeline/outputs/

## 目录结构

- wp0_freeze_assets: 冻结基线预测资产
- wp1_umap: 导出融合特征并绘制 UMAP 双图
- wp2_stratified_robustness: MW/TPSA 分层鲁棒性评估
- wp3_activity_cliff: 活性悬崖候选检索 + 构象注意力导出
- wp4_ablation: 四视图消融 CV 与图表
- wp5_stats_calibration: 统计汇总与校准曲线
- common: 通用工具函数

## 快速执行

在 TV-STFN 根目录运行：

bash tvstfn_paper_pipeline/run_all.sh

默认是 fast 模式（跳过高算力消融 WP4，优先产出高价值图表）。

如果你要完整全跑（包含 WP4 消融）：

MODE=full bash tvstfn_paper_pipeline/run_all.sh

## 智能调度器（推荐）

支持断点续跑、按输出自动跳过、只跑指定步骤：

python tvstfn_paper_pipeline/run_pipeline.py --mode fast --resume

只跑某一步（例如 WP1 出图）：

python tvstfn_paper_pipeline/run_pipeline.py --mode fast --resume --only wp1_export_embeddings,wp1_plot_umap

全量模式（含 WP4 消融）：

python tvstfn_paper_pipeline/run_pipeline.py --mode full --resume

WP4 将自动调用“空闲 GPU 实时调度器”，哪个 GPU 空出来就自动派发一个消融分片，直到 5 个变体全部完成并自动合并。

你也可以单独启动调度器：

python tvstfn_paper_pipeline/wp4_ablation/dispatch_ablation_free_gpus.py \
  --root /data/workplace/jwx/TV-STFN \
  --python-bin python \
  --out-dir tvstfn_paper_pipeline/outputs/wp4_ablation \
  --variants full,wo_0d,wo_1d,wo_2d,wo_3d

查看将执行哪些命令（不实际运行）：

python tvstfn_paper_pipeline/run_pipeline.py --mode fast --dry-run

若指定解释器：

PYTHON_BIN=/path/to/python bash tvstfn_paper_pipeline/run_all.sh

## 自动化测试（低算力烟雾测试）

在 TV-STFN 根目录运行：

bash tvstfn_paper_pipeline/tests/run_tests.sh

测试内容：

- WP1 的 UMAP 绘图脚本最小样本可执行
- WP5 的统计与校准脚本最小样本可执行

说明：该测试不触发模型重训练，主要用于快速验证脚本链路与输出文件完整性。

## 单独运行示例

1) UMAP：

python tvstfn_paper_pipeline/wp1_umap/export_embeddings.py \
  --data-dir tetraview_processed \
  --weights best_tetraview_model.pth \
  --pred-dir benchmark_results \
  --out-dir tvstfn_paper_pipeline/outputs/wp1_umap

python tvstfn_paper_pipeline/wp1_umap/plot_umap.py \
  --npz tvstfn_paper_pipeline/outputs/wp1_umap/umap_embeddings.npz \
  --out-dir tvstfn_paper_pipeline/outputs/wp1_umap

2) 分层鲁棒性：

python tvstfn_paper_pipeline/wp2_stratified_robustness/stratified_eval.py \
  --pred-dir benchmark_results \
  --out-dir tvstfn_paper_pipeline/outputs/wp2_stratified

3) 活性悬崖：

python tvstfn_paper_pipeline/wp3_activity_cliff/find_activity_cliffs.py \
  --pred-dir benchmark_results \
  --out-dir tvstfn_paper_pipeline/outputs/wp3_cliff

python tvstfn_paper_pipeline/wp3_activity_cliff/export_conformer_attention.py \
  --indices-csv tvstfn_paper_pipeline/outputs/wp3_cliff/cliff_shortlist_top10.csv \
  --data-dir tetraview_processed \
  --weights best_tetraview_model.pth \
  --out-dir tvstfn_paper_pipeline/outputs/wp3_cliff

4) 四视图消融：

python tvstfn_paper_pipeline/wp4_ablation/run_ablation_cv.py \
  --data-dir tetraview_processed \
  --out-dir tvstfn_paper_pipeline/outputs/wp4_ablation \
  --n-folds 5 --epochs 25

5) 统计与校准：

python tvstfn_paper_pipeline/wp5_stats_calibration/stats_and_calibration.py \
  --pred-dir benchmark_results \
  --out-dir tvstfn_paper_pipeline/outputs/wp5_stats \
  --focus-model TVSTFN

## 基于结果反哺训练（增强版）

1) 先从 WP2/WP3 结果生成难样本权重：

python tvstfn_paper_pipeline/optimization/build_hard_sample_weights.py \
  --cliff-csv tvstfn_paper_pipeline/outputs/wp3_cliff/cliff_shortlist_top10.csv \
  --smiles-csv CycPeptMPDB_Peptide_PAMPA.csv \
  --out-csv tvstfn_paper_pipeline/outputs/optimization/sample_weights_v1.csv

2) 在主训练脚本中启用权重采样：

python benchmark_tvstfn_fast.py \
  --n-folds 5 \
  --epochs 30 \
  --sample-weights-csv tvstfn_paper_pipeline/outputs/optimization/sample_weights_v1.csv \
  --sample-weight-power 1.2 \
  --tag hard_v1

说明：

- benchmark_tvstfn_fast.py 已支持 sample-weights-csv 与 sample-weight-power 参数。
- fold 预测文件已自动写入 __dataset_index，后续 WP3 对齐会更稳定。

## 依赖

除项目原依赖外，建议安装：

- umap-learn
- matplotlib
- scikit-learn
- rdkit

如果未安装 umap-learn，脚本会自动退化为 TSNE。