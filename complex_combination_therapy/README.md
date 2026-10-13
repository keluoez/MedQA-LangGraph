# BioReasonPipeline - 联合治疗模块

## 数据类别
**Complex-Combination Therapy**: 联合治疗的协同机制

## 文件
- `pipeline.py` - 主程序
- `combination_pairs.json` - 输入数据
- `output/batch_output.json` - 输出结果

## 运行
```bash
python pipeline.py --count N
```

## 输出格式
```json
{
  "question": "What is the synergistic mechanism of Aspirin + Clopidogrel in Antithrombotic therapy?",
  "answer": "...",
  "category": "Complex-Combination Therapy",
  "drugs": "Aspirin + Clopidogrel",
  "scene": "Antithrombotic therapy",
  "targets": ["PTGS1", "P2RY12"],
  "uniprot_id": "P23219/Q9H0D6"
}
```

## 数据库
- **PubMed** - 协同机制文献
- **UniProt** - 靶点蛋白信息
