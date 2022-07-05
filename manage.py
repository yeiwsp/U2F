#!/usr/bin/env python
"""Django's command-line utility for administrative tasks1111111111111111111."""
import os
import sys

#获取文件的绝对路径
this = os.path.abspath(os.path.dirname(__file__))
module = os.path.split(this)[0]
#print(this,module)
sys.path.append(module)


def main():
    #指定Django使用的系统配置文件
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'software_match.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
    #运行execute

if __name__ == '__main__':
    main()
