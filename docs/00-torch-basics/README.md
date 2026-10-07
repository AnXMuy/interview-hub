# PyTorch 基础

按 [torch-basics.md](torch-basics.md) 顺序阅读：张量操作 → 计算图 → 模型训练 → 常见错误；末尾有面试问答。所有示例在本地 Python + PyTorch 即可运行。

## 练习

完成 `solutions/torch_basics.py` 中的 `masked_mean`，运行 `python -m grader 00-torch-basics`。接口及诊断说明见 [测评说明](../../grader/README.md)。

## 高频面试问答

**问：`view` 和 `reshape` 有什么区别？**

答：`view` 要求底层内存布局兼容；`reshape` 必要时会复制，不能依赖它与原张量共享存储。

**问：`model.eval()` 会关闭梯度吗？**

答：不会，它只影响 Dropout、BatchNorm 等模块；推理时另用 `torch.inference_mode()`。
