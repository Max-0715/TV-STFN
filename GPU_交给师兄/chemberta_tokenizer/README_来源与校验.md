# chemberta_tokenizer/ — 离线 ChemBERTa 分词器（随脚本一起发）

来源仓库：`seyonec/ChemBERTa-zinc-base-v1`（HuggingFace）
固定 revision：`761d6a18cf99db371e0b43baf3e2d21b3e865a20`（lastModified 2021-05-20）
下载日期：2026-09-14

| 文件 | sha256 |
| --- | --- |
| config.json | a484ab1497bf438b2dcbda35c6ce518ca8043539ff1a56d713aa84bcf2cba225 |
| vocab.json | 791f9dd0e7dacb628c2f3155671a702044b1da537784eaa5663fb36ee7b8d42f |
| merges.txt | f8c368d3b2f44cf4eec62aa576b69945302fcfa8eba065048d3725259671f2bb |
| tokenizer_config.json | 36557a021e64d222c7ff0fef5369bc87ae6e0780264af0f35d8c6461cf106577 |
| special_tokens_map.json | 7638f5bbbe86ef6d604ef28ad3647dc690d6d117c81c0d63e885416be8da1150 |

说明：

* 分词器本身不需要模型权重（`pytorch_model.bin`），本目录只放分词所需的 5 个文件（约 13 KB）。
* 上游仓库没有 `tokenizer.json`，`AutoTokenizer` 会从 `vocab.json` + `merges.txt` 构建
  RoBERTa(byte-level BPE)，vocab_size = 767，`<pad>` = 1。
* 一致性已验证（见 `../02_补丁_离线tokenizer_修复_2026-09-14.md`）：用本目录复现
  `tetraview_processed/data_*.pt` 里存档的 `view2_1d`，token id 与 attention mask 与旧
  PAMPA 张量逐位相同。
* 不要改名、不要删其中任何一个文件，否则 `preprocess_assay.py --check-tokenizer` 会报错。
