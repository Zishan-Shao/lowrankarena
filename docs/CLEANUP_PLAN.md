# eval_encoder 清理计划

## 要删除的文件/目录

### 1. 重复的目录
- `eval_encoder/eval_encoder/` - 误创建的重复目录
- `eval_encoder/kernels/` - 内容与上层 kernels/encoder_kernels/ 重复

### 2. 旧的/临时的脚本和工具
- `finetune_compressed_correct.py` - 已被 glue_pipeline.py 替代
- `test_all_tasks_verification.sh` - 旧测试脚本
- `test_dense_all_tasks.sh` - 旧测试脚本
- `test_memory_cleanup.py` - 临时测试文件
- `test_memory_difference.sh` - 临时测试脚本
- `check_fixes_applied.sh` - 临时检查脚本
- `verify_dependencies.py` - 与 check_dependencies.py 重复

### 3. 旧文档（可归档）
- `MEMORY_FIX_SUMMARY.md` - 归档到 archived/

### 4. 空目录
- `logs/` - 如果为空
- `finetuned_models/` - 如果为空
- `tools/` - 需要检查内容

## 保留的核心文件
- `run_encoder_benchmark.py` - 主评测脚本
- `glue_pipeline.py` - GLUE pipeline
- `blocks.py` - SVD block 实现
- `flashsvd_backend.py` - FlashSVD 后端
- `load_compressed_model.py` - 模型加载工具
- `check_dependencies.py` - 依赖检查
- `README.md`, `CHANGELOG.md`, `DEPENDENCIES.md` - 文档
- `scripts/` - 脚本目录
- `docs/` - 文档目录
- `archived/` - 已归档文件
