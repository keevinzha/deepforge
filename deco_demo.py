# -*- coding: utf-8 -*-
"""
@Time ： 2025/10/27 17:12
@Auth ： keevinzha
@File ：deco_demo.py
@IDE ：PyCharm
"""
# python
def deco(cls):
    print(f"decorator called for {cls.__name__}")
    return cls

@deco
class A:
    print("class body executed for A")
    def __init__(self):
        print("A.__init__ called")

print("After class definition")

# 实例化发生在这里
a = A()
print("After instantiation")
