
l = [(m1 + 0.2*m1)/(m2- 0.2*m2) - (m1 - 0.2*m1)/(m2 + 0.2*m2) for m1 in range(1,101) for m2 in range(1,101)]

# l = [(m1 + 0.2 * m1) / (m2 - 0.2 * m2) - (m1 - 0.2 * m1) / (m2 + 0.2 * m2) for m1 in range(1, 101) for m2 in
#      range(1, 101)]
#
print(min(l))
print(max(l))
#
# print(max(l) / 100)

print(max(l)/100)


class A:
    def __init__(self, idf):
        print("init A")
        self.id = idf

    def fun(self):
        print("funkcja z A")


class B(A):
    def funB(self):
        print("funckja z B")


class C(A):
    def funC(self):
        print("funckja z C")


x = A(1)
x.fun()

x.__class__ = B
x.funB()

x.__class__ = C
x.funC()