l = [(m1 + 0.2 * m1) / (m2 - 0.2 * m2) - (m1 - 0.2 * m1) / (m2 + 0.2 * m2) for m1 in range(1, 101) for m2 in
     range(1, 101)]

print(min(l))
print(max(l))

print(max(l) / 100)
