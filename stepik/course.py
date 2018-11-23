from sys import stdin, stdout
import re

stream = None
try:
    stream = open('file.txt', 'r')
except:
    stream = stdin

input = int(stream.readline())
answer = ''