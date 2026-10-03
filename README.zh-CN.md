# 论文复现材料：多说话人转写中的精确重复抑制审计

对应论文 **Auditing Exact-Repetition Containment in Multi-Speaker Transcription**
的原始 `tmm_revision_v003` 保存版本；2026-10-03 新增了对应 `tmm_revision_v013`
的 token 4-gram 实验。[English README](README.md)

本仓库提供精确重复处理算法、字符与 Whisper token 对照方法、整场会议
cpCER 评分器、冻结的解码源码、合成测试，以及论文中的数值结果与来源哈希。
默认参数为 `(12, 6, 24, 2)`，分别表示最大周期、最少重复次数、最短重复跨度、
保留次数。算法独立处理每个转写片段，不读取参考转写。

## 运行

使用 Python 3.11 或更新版本，在仓库根目录执行：

```sh
python code/verify_manifest.py
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python code/verify_public_tables.py
python code/verify_extended_results.py
python code/verify_ngram_manifest.py
python code/verify_ngram_results.py
python -B -m unittest discover -s code/tests -p test_analyze_ngram_postprimary_v1.py -v
python code/run_guard.py --text abcdabcdabcdabcdabcdabcd
```

最后一条命令输出 `abcdabcd`。数值核验覆盖原始 144 行会议评分、12 个配对
bootstrap 对比、616 行解码对照评分、48 个直接基线汇总单元、864 行删除对照，
以及 80,256 条方法记录。未安装 PyTorch 时会明确跳过三个可选解码张量测试；
安装 `requirements-decoder.txt` 后可运行全部 23 项测试。

[代码使用说明](code/README.md)提供合成 JSONL 示例和输入格式；
[论文证据映射](provenance/paper_evidence_map.json)对应论文表格与具体结果文件。
[结果目录](results/)、[图表目录](figures/)、[协议](protocol/)和
[验证记录](environment/VALIDATION.json)保留了复核依据。

## 可复现范围

新增 [n-gram 实验](NGRAM_REPRODUCIBILITY.md)覆盖 Primary/public 的 18 段完整录音、
1,744 个窗口和 222 个原始批次。A/B/C/D 分别为 cap+fallback、再加 guard、
cap+fallback+4-gram blocking，以及 blocking 后再加 guard；micro cpCER 为
1.073020 / 1.068336 / 1.110009 / 1.110009。Blocking 相对 A 增加 3.6989 个百分点，
95% 配对区间为 [0.5292, 7.5640]；字符残余事件从 4 降为 0，D 与 C 文本一致。
本次未扩展至 Replication 或 n=8，旧听审标签不用于判断 C/D 的语义安全。
新增验证包括 72 行录音计数、四组配对对比、50,000 次 bootstrap 与 15 项合成测试。

可直接运行文本算法、合成测试、会议评分及公开数值复核。原始音频实验的完整
重跑仍需自行取得许可语料、公共模型权重、MVDR 输出波形、批次映射和执行记录。
私有 S-FULLTAC 前端权重及完整前端重建流程未发布，因此本仓库不能单独还原
全部历史波形与识别输出。公开解码协议经过路径清理，其哈希不同于原执行封存件。

仓库不包含语料音视频、逐字语料转写、模型权重、个人听审表或审阅者身份。
听审仅发布汇总结果；概率抽样扩展缺少播放别名到音频的校验映射，另一次残余
事件审计覆盖 41 个窗口中的 43 次操作。数据和模型的官方来源见
[第三方说明](THIRD_PARTY_NOTICES.md)。

cpCER 可大于 1，全部输出为空时为 1.0。分数降低和精确重复消失，并不单独证明
保留语音的语义安全或整段转写可用性。原始代码和自主编写文档使用 [MIT 许可证](LICENSE)，
第三方数据、模型和依赖仍遵循各自条款。
