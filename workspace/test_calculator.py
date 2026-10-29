# 导入我们需要测试的 add() 函数。
from calculator import add


# pytest 会自动发现 test_ 开头的测试函数。
def test_add():

    # 2 + 3 的正确结果应该是 5。
    # 但 calculator.py 当前执行的是减法。
    # 因此第一次测试应该失败。
    assert add(2, 3) == 5