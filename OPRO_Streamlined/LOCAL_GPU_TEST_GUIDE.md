# OPRO本地GPU测试指南

## 快速开始

### 1. 检查GPU状态
```bash
python run_opro.py --gpu-check
```
这会显示：
- GPU可用性
- GPU内存大小
- 推荐的模型
- 已安装的包状态

### 2. 测试模型加载
```bash
python run_opro.py --mode test-model
```
这会：
- 尝试加载最适合你GPU的Llama模型
- 运行一个简单的生成测试
- 显示GPU内存使用情况

### 3. 快速测试运行（推荐开始）
```bash
python run_opro.py --mode test-run
```
这会：
- 运行2次迭代的优化测试
- 使用较小的配置参数
- 验证整个系统是否正常工作

### 4. 完整优化运行
```bash
python run_opro.py --mode optimize
```
这会运行完整的OPRO优化过程

## 运行模式说明

| 模式 | 用途 | 时间 | 推荐场景 |
|------|------|------|----------|
| `--gpu-check` | 检查GPU状态 | 几秒 | 第一次使用 |
| `test-model` | 测试模型加载 | 1-3分钟 | 验证模型工作 |
| `test-run` | 快速测试优化 | 3-5分钟 | 验证系统工作 |
| `optimize` | 完整优化 | 10-30分钟 | 生产使用 |

## GPU内存要求

| GPU内存 | 推荐模型 | 性能 |
|---------|----------|------|
| >= 16GB | Llama-3-8B (全精度) | 最佳 |
| >= 10GB | Llama-3-8B (4bit量化) | 很好 |
| >= 6GB  | Llama-3.2-3B (4bit量化) | 良好 |
| < 6GB   | Llama-3.2-1B | 基础 |

## 常见问题解决

### 1. 内存不足错误
```
GPU out of memory!
```
**解决方案**：
- 系统会自动清理缓存并重试
- 如果仍然失败，会使用更小的生成长度
- 考虑使用更小的模型

### 2. 模型下载失败
```
Error loading meta-llama/Meta-Llama-3-8B-Instruct
```
**解决方案**：
- 确保有网络连接
- 可能需要Hugging Face登录：`huggingface-cli login`
- 系统会自动尝试更小的模型

### 3. 量化库缺失
```
bitsandbytes: NOT INSTALLED
```
**解决方案**：
```bash
pip install bitsandbytes
```

## 输出文件

成功运行后，会生成：
- `prompts/optimized_prompt.txt` - 优化后的提示词
- `prompts/optimization_history.json` - 优化历史记录
- `config/test_config.json` - 测试配置（仅test-run模式）

## 性能监控

运行时会显示：
- GPU内存使用情况
- 模型加载时间
- 每次迭代的得分改进
- 总优化时间

## 建议的测试流程

1. **首次使用**：
   ```bash
   python run_opro.py --gpu-check
   python run_opro.py --mode test-model
   python run_opro.py --mode test-run
   ```

2. **日常使用**：
   ```bash
   python run_opro.py --mode optimize
   ```

3. **调试问题**：
   ```bash
   python run_opro.py --mode info
   ```

这样你就可以在本地GPU环境中高效地测试和运行OPRO系统了！ 