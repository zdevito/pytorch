import sys
print("I AM ALIVE!")
print(id(None))
sys.path.append('/private/home/zdevito/miniconda3/envs/py38/lib/python3.8/site-packages/')
import regex
print("regex imported, running tests...")
from unittest import main
main(exit=False, module='test_regex')
