# ICML/NeurIPS/ICLR 2025 论文关键词总结

## 运行方式

安装依赖:

```bash
pip install -r requirements/requirements_icml_neurips_iclr_2025.txt
```

从 0 抓取 + 分析 + 出图:

```bash
python3 scripts/icml_neurips_iclr_2025_keyword_report.py \
  --output-dir outputs/conference_2025_keyword_report
```

抓取扩展会议（在三会基础上增加 MIDL + EMNLP）:

```bash
python3 scripts/icml_neurips_iclr_2025_keyword_report.py \
  --include-related \
  --output-dir outputs/conference_2025_keyword_report_extended
```

抓取网络会议（SIGCOMM / NSDI / MobiCom / INFOCOM，DBLP 源）:

```bash
python3 scripts/icml_neurips_iclr_2025_keyword_report.py \
  --include-network \
  --output-dir outputs/conference_2025_keyword_report_network_only
```

网络会议仅抓标题（禁用 OpenAlex 摘要回填，速度更快）:

```bash
python3 scripts/icml_neurips_iclr_2025_keyword_report.py \
  --include-network \
  --disable-openalex-abstracts \
  --output-dir outputs/conference_2025_keyword_report_network_only
```

仅抓取扩展会议（不重复抓 ICLR/NeurIPS/ICML）:

```bash
python3 scripts/icml_neurips_iclr_2025_keyword_report.py \
  --include-related \
  --exclude-core \
  --output-dir outputs/conference_2025_keyword_report_related_only
```

追加自定义会场（可重复）:

```bash
python3 scripts/icml_neurips_iclr_2025_keyword_report.py \
  --include-related \
  --extra-group "MyConf:SomeOrg/2025/Conference" \
  --output-dir outputs/conference_2025_keyword_report_extended
```

基于已有 `papers_2025.json` 仅重算统计/图表:

```bash
python3 scripts/icml_neurips_iclr_2025_keyword_report.py \
  --papers-json outputs/conference_2025_keyword_report/papers_2025.json \
  --output-dir outputs/conference_2025_keyword_report
```

启用浏览器模拟环境（可选，需安装 Playwright 与浏览器）:

```bash
python3 scripts/icml_neurips_iclr_2025_keyword_report.py \
  --output-dir outputs/conference_2025_keyword_report \
  --use-playwright
```

## 输出文件

- `outputs/conference_2025_keyword_report/papers_2025.json`
  - 逐篇论文数据，包含标题、摘要、关键词、主题、会议、轨道等字段
- `outputs/conference_2025_keyword_report/summary_2025.json`
  - 各会议 Top 关键词、主题分布、跨会趋势（延续热点 / 新兴趋势）
- `outputs/conference_2025_keyword_report/report_2025.md`
  - 简要文本报告
- `outputs/conference_2025_keyword_report/figures/*.png`
  - 图表输出（Top 关键词、热力图、主题分布）
