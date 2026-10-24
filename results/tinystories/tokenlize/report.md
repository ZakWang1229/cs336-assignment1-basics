# TinyStories tokenizer 运行报告

运行日期：2026-10-22（Asia/Shanghai）。

## 运行范围

执行本目录的 `test.py`，未修改脚本的实验逻辑。使用 `../train/vocab.pkl`、`../train/merges.pkl` 和 `../train/summary.json` 加载 TinyStories tokenizer：词表大小 10,000，合并数 9,743，特殊 token 为 `<|endoftext|>`。

输入为完整 TinyStories 验证集，以及从 OpenWebText 验证集随机抽取的 4,288 篇完整文档（随机种子 336）。两份输入都使用同一个 TinyStories tokenizer。编码使用 `encode` 一次性处理文本，随后保存 NumPy 数组并解码校验。

## 实测结果

压缩率采用加载后的文本重新编码为 UTF-8 的字节数除以 token 数；脚本原始输出的 `len(str)` 是字符数，不能直接作为字节数。

| 输入 | 字符数 | UTF-8 字节数 | Token 数 | Bytes/token | 解码与原文相等 |
| --- | ---: | ---: | ---: | ---: | --- |
| TinyStories 验证集 | 22,493,387 | 22,502,601 | 5,465,883 | 4.1169 | 是 |
| OpenWebText 随机样本 | 20,828,217 | 21,007,176 | 6,565,621 | 3.1996 | 是 |

该 TinyStories tokenizer 在这份 OpenWebText 样本上的 bytes/token 更低，即相同字节数的文本通常需要更多 token。两份样本的规模与抽样方式不同，该比较描述本次输入，不能当作所有文本上的总体结论。

## 保存结果

| 文件 | 类型 | 元素数 | 文件字节数 |
| --- | --- | ---: | ---: |
| `tiny_story_valid_tokens.npy` | `uint16` | 5,465,883 | 10,931,894 |
| `owt_valid_sample_20MiB_tokens.npy` | `uint16` | 6,565,621 | 13,131,370 |

保存后的两个数组均重新以只读内存映射加载检查。两者 token ID 最小值均为 10，最大值均为 9,999，未超出 `uint16` 范围。表内文件大小包含 NumPy 文件头。

## 耗时与资源

用外层 Python 启动器的 `time.perf_counter` 统计墙钟时间，使用 `resource.getrusage(resource.RUSAGE_SELF)` 记录当前进程的 CPU 时间和峰值 RSS。

| 指标 | 实测值 |
| --- | ---: |
| 总墙钟时间 | 79.881853 秒 |
| 用户态 CPU 时间 | 87.307444 秒 |
| 内核态 CPU 时间 | 0.311612 秒 |
| 平均 CPU 使用率（一个核心为 100%） | 109.69% |
| 峰值 RSS | 764,420 KiB（746.50 MiB） |
| 两份输入 UTF-8 字节数之和 | 43,509,777 |
| 全流程处理速率 | 544,677 bytes/s（约 0.545 MB/s） |

计时包含导入、加载 tokenizer、读取两份文本、编码、数组转换和保存、解码及比较，不是纯编码吞吐量。CPU 时间覆盖启动器进程生命周期，与墙钟计时边界略有差异。峰值内存还包含同时保留的文本、Python token 列表、数组转换和解码结果，不能用于判断 `encode_iterable` 的内存效率。

本次没有分别计时两个数据集的编码阶段，也没有运行 OpenWebText 32K tokenizer、训练集全量编码或 Pile 编码，因此不提供这些结果或纯编码耗时外推。

原始标准输出及资源记录见 [`run.log`](run.log)。
