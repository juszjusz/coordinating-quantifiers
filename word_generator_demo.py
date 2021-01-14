import random
f = open(r'sorted_syllables.txt')
syllables = f.read().splitlines()
f.close()
# generate random
for _ in range(0, 100):
    word_syllables_size = random.randint(2, 3)
    print(''.join(random.choice(syllables) for _ in range(word_syllables_size)))



