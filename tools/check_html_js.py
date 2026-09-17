# -*- coding: utf-8 -*-
"""提取 index.html 里的 <script> 存成 .js，交给 node --check 做语法校验。"""
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
html = open(os.path.join(BASE, "index.html"), encoding="utf-8").read()
m = re.findall(r"<script>(.*?)</script>", html, re.S)
print(f"找到 {len(m)} 段 script")
js = "\n".join(m)
out = os.path.join(BASE, "_index_check.js")
open(out, "w", encoding="utf-8").write(js)
print(f"写出 {out}（{len(js)} 字符）")
print("relabel 处理器:", "有" if "relabelBubble" in js else "**缺失**")
print("relabel 消息分支:", "有" if "msg.type === 'relabel'" in js else "**缺失**")
