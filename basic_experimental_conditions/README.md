# BioReasonPipeline - 实验条件模块

## 数据类别
**Basic-Experimental Conditions**: 实验条件对蛋白活性的影响

## 文件
- `pipeline.py` - 主程序
- `experiment_pairs.json` - 输入数据
- `output/batch_output.json` - 输出结果

## 运行
```bash
python pipeline.py --count N
```

## 输出格式
```json
{
  "question": "How does Low pH environment (pH 1-2) affect Pepsin activity and function?",
  "answer": "...",
  "category": "Basic-Experimental Conditions",
  "condition": "Low pH environment (pH 1-2)",
  "protein": "Pepsin",
  "uniprot_id": "P00790"
}
```

## 数据库
- **UniProt** - 蛋白质功能信息
- **PubMed** - 实验条件效应文献
