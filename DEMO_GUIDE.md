# CBT + ICD-11 系统功能演示指南

本指南将帮助您演示和测试CBT (认知行为治疗) 与ICD-11医学知识系统的集成功能。

## 📋 准备工作

### 1. 确保系统依赖已安装

```bash
# 基础依赖
pip install torch transformers langchain sentence-transformers faiss-cpu
pip install fastapi uvicorn requests beautifulsoup4 numpy pandas

# CBT系统依赖
pip install -r CBT_System/requirements.txt
```

### 2. 数据准备

#### CBT数据收集与处理
```bash
# 进入CBT系统目录
cd CBT_System

# 运行自动化设置（推荐）
python setup.py

# 或者手动步骤：
# python data_collector.py    # 收集CBT数据
# python data_processor.py    # 处理数据
# python vectorizer.py        # 生成向量索引
```

#### ICD-11数据准备
```bash
# 获取ICD-11数据
python fetch_icd11_ch6.py

# 生成向量索引
python generate_faiss_index.py
```

## 🚀 演示文件说明

### 1. `quick_demo.py` - 快速功能演示 (推荐)

**适用场景**: 快速验证系统功能，适合现场演示
**运行时间**: 2-5分钟

```bash
python quick_demo.py
```

**演示内容**:
- ✅ CBT系统状态检查
- 🔍 CBT相关性检测
- 🔎 CBT技术搜索
- 💡 CBT推荐系统
- 🏥 ICD-11知识检索
- 💬 集成对话流程

### 2. `test_cbt_icd11_demo.py` - 完整功能测试

**适用场景**: 详细测试和开发调试
**运行时间**: 10-20分钟

```bash
python test_cbt_icd11_demo.py
```

**测试选项**:
1. 完整测试 (推荐)
2. 仅测试CBT系统
3. 仅测试ICD-11系统
4. 仅测试集成对话
5. 仅测试API端点
6. 生成完整测试报告

## 🎯 演示场景

### 场景1: CBT技术推荐演示

**用户输入示例**:
- "我感到很焦虑，不知道该怎么办"
- "如何应对负面思维？"
- "我总觉得自己什么都做不好"

**展示功能**:
- 智能识别心理健康问题
- 推荐相应的CBT技术
- 提供专业的指导建议

### 场景2: ICD-11医学知识检索

**搜索示例**:
- "抑郁症的诊断标准"
- "焦虑障碍的分类"
- "创伤后应激障碍"

**展示功能**:
- 语义相似度搜索
- 医学知识精准检索
- 专业诊断信息提供

### 场景3: 集成对话流程

**模拟对话**:
```
用户: "我最近总是感到很焦虑，晚上睡不着觉，心跳加速"

系统处理流程:
1. 情感分析: 检测到焦虑情绪
2. ICD-11检索: 查找焦虑障碍相关信息
3. CBT分析: 识别需要放松技术和认知重构
4. 生成回复: 提供专业建议和具体技术
```

## 🌐 Web界面演示

### 启动完整系统

```bash
# 1. 启动后端API服务器
python fastapi_server.py

# 2. 启动前端服务 (新终端)
cd web
npm install
npm run dev
```

### 访问界面

- **后端API**: http://localhost:8000
- **前端界面**: http://localhost:3000
- **API文档**: http://localhost:8000/docs

### Web演示功能

1. **多语调模式**: 专业、共情、简洁等不同回复风格
2. **实时对话**: 即时AI回复和CBT技术推荐
3. **反馈收集**: 用户评分和反馈，用于系统优化
4. **对话历史**: 保持上下文连贯性

## 📊 系统健康检查

### API健康检查

```bash
curl http://localhost:8000/health
```

**期望输出**:
```json
{
  "status": "healthy",
  "psychologist_llm_loaded": true,
  "store_loaded": true,
  "device": "cuda",
  "cbt_available": true,
  "cbt_techniques": 12,
  "interactions_count": 45
}
```

### CBT系统状态

```bash
curl http://localhost:8000/api/cbt/status
```

## 🔧 故障排除

### 常见问题

1. **CBT系统不可用**
   ```bash
   cd CBT_System
   python setup.py
   ```

2. **ICD-11索引不存在**
   ```bash
   python generate_faiss_index.py
   ```

3. **依赖模块缺失**
   ```bash
   pip install -r requirements.txt
   pip install -r CBT_System/requirements.txt
   ```

4. **CUDA内存不足**
   - 在配置中降低batch_size
   - 使用CPU模式: 设置CUDA_VISIBLE_DEVICES=""

### 日志文件

- CBT处理日志: `CBT_System/cbt_data/processing.log`
- API服务器日志: 控制台输出
- 交互记录: `interactions.json`

## 📈 性能优化建议

### 硬件配置

**最低配置**:
- CPU: 4核心
- 内存: 8GB RAM
- 存储: 10GB 可用空间

**推荐配置**:
- GPU: NVIDIA GTX 1060 或更高
- CPU: 8核心
- 内存: 16GB RAM
- 存储: SSD 20GB

### 软件优化

1. **使用GPU加速**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **模型量化** (如需要)
   - 参考 `OPRO_Streamlined/` 中的量化脚本

3. **缓存优化**
   - 预加载模型和索引
   - 使用Redis进行会话缓存

## 📝 演示脚本模板

### 5分钟快速演示

```text
1. 介绍系统 (30秒)
   - CBT + ICD-11 集成心理健康咨询系统
   - 循证医学 + AI技术

2. CBT功能演示 (2分钟)
   - 运行: python quick_demo.py
   - 选择CBT演示部分
   - 展示技术推荐和搜索

3. ICD-11演示 (1.5分钟)
   - 展示医学知识检索
   - 语义搜索能力

4. 集成对话 (1分钟)
   - 模拟真实用户场景
   - 展示智能回复生成
```

### 15分钟详细演示

```text
1. 系统架构介绍 (3分钟)
2. CBT系统详细功能 (4分钟)
3. ICD-11系统详细功能 (3分钟)
4. Web界面实时演示 (3分钟)
5. Q&A (2分钟)
```

## 🔗 相关资源

- **CBT系统文档**: `CBT_System/README.md`
- **OPRO优化系统**: `OPRO_Streamlined/README.md`
- **API文档**: 启动服务器后访问 `/docs`
- **前端组件**: `web/src/components/`

## 📞 技术支持

如果在演示过程中遇到问题:

1. 检查系统日志文件
2. 确认所有依赖已正确安装
3. 验证数据文件是否完整
4. 查看故障排除章节

---

**祝您演示成功！** 🎉

此系统展示了现代AI技术在心理健康领域的应用潜力，结合了循证医学知识和智能对话技术，为用户提供专业、个性化的心理健康支持。 