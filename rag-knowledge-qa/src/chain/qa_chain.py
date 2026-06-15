"""
问答链 — 单轮 RAG 问答。
支持 stuff / refine / map_reduce 三种链类型：
- stuff: 将所有检索文档一次性塞入 prompt（适合文档量少时）
- refine: 逐篇精炼回答（适合需要综合多篇信息时）
- map_reduce: 先对各文档独立生成，再汇总（适合大量文档时）
"""
