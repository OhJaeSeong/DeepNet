"""
Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

테스트 함수를 실행한다.
"""

import sys
import traceback


class TestMain:
    """TestMain 클래스."""

    test_count = 0
    error_count = 0

    def __init__(self):
        """생성자."""
        pass

    @staticmethod
    def run(func):
        """시험 함수를 실행한다."""

        TestMain.test_count += 1

        print(f'\x1B[36m[Test {TestMain.test_count}] {func.__name__}\x1B[37m')

        try:
            func()
        except (ArithmeticError, AssertionError, EOFError, OSError, RuntimeError, SystemError, ValueError) as e:
            TestMain.error_count += 1

            print('\x1B[31m', end='', file=sys.stderr)
            traceback.print_exc()
            print('\x1B[37m', end='', file=sys.stderr)

    @staticmethod
    def report():
        """결과를 출력한다."""

        print(
            f'\x1B[36mTotal = {TestMain.test_count}\n'
            f'Error = {TestMain.error_count}\x1B[37m')


def run(func):
    """시험 함수를 실행한다."""

    TestMain.run(func)


def report():
    """결과를 출력한다."""

    TestMain.report()


def log(msg):
    """로그 메시지를 출력한다."""

    print(f'\x1B[33m{msg}\x1B[37m')
