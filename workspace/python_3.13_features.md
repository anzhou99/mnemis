# Python 3.13 核心新特性技术笔记

## 1. 自由线程支持（PEP 703）

### 核心功能
移除全局解释器锁（GIL），实现真实的多核心并行执行。从 Python 3.14 开始将成为官方支持功能（不再是实验性）。

### 作用场景
- 数据密集型任务
- CPU 密集型任务（如科学计算、并行处理）

### 代码示例
使用标准 `threading` 模块编写多线程程序，无需特殊语法即可享受多核心性能：

```python
import threading

def compute_intensive_task():
    # CPU密集型计算
    result = 0
    for i in range(10**7):
        result += i
    return result

# 创建多个线程
threads = []
for _ in range(4):
    t = threading.Thread(target=compute_intensive_task)
    threads.append(t)
    t.start()

# 等待所有线程完成
for t in threads:
    t.join()
```

---

## 2. 即时编译器（PEP 744）

### 核心功能
运行时将 Python 代码编译为机器码，提升执行效率。当前为实验性功能。

### 作用场景
- 循环密集型代码
- 需要加速的热点路径

### 使用方法
通过命令行启用：
```bash
python -X jit your_script.py
```

或设置环境变量：
```bash
PYTHONJIT=1 python your_script.py
```

### 代码示例
常规 Python 脚本无需修改即可受益：

```python
def hot_loop():
    result = 0
    # 这个循环会被JIT编译优化
    for i in range(10**8):
        result += i * i
    return result

print(hot_loop())
```

---

*注：本文档基于 Python 3.13 实验版本编写，部分功能在正式发布时可能有所调整。*
