from sys import stdin

stream = None
try:
    stream = open('input.txt', 'r')
except:
    stream = stdin

n = int(stream.readline())


# def create(namespace, parent):
#     print("CREATE", namespace, parent)
#
#
#
# def add(namespace, var):
#     print("ADD", namespace, var)
#     temp = namespaces.get(namespace, [])
#     temp.append(var)
#     namespaces[namespace] = temp
#
#
# def get(namespace, var):
#     print("GET", namespace, var)
#
#
# commands = {"create": create, "add": add, "get": get}
# namespaces = {"global": []}
#
# for i in range(n):
#     raw = [item.strip() for item in stream.readline().split(" ")]
#     command = commands.get(raw[0])
#     command(raw[1], raw[2])
#
# print(namespaces)


def recursive(ancestor, child):
    ancestors = memory.get(child, [])
    if ancestor == child:
        return True
    for item in ancestors:
        if item == ancestor:
            return True
    for item in ancestors:
        if recursive(ancestor, item):
            return True


memory = {}

for i in range(n):
    raw = [item.strip() for item in stream.readline().split(" ")]
    child = raw[0]
    memory[child] = []
    for item in raw[1:]:
        if item == ":":
            continue
        memory[child].append(item)

n = int(stream.readline())
for i in range(n):
    ancestor, child = map(str.strip, stream.readline().split(" "))
    print("Yes") if recursive(ancestor, child) else print("No")
