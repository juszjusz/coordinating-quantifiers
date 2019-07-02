from objects.perception import Stimulus
from objects.agent import Agent

a = Agent(1)
print(a.language.lxc.shape)

c = a.discriminate([Stimulus(1, 2), Stimulus(3, 4)], 0)
if c is None:
    print("discrimination failed")
else:
    print(c)

print(a.language.lxc.shape)

c = a.discriminate([Stimulus(1, 2), Stimulus(3, 4)], 1)
if c is None:
    print("discrimination failed")
else:
    print(c)

c = a.discriminate([Stimulus(1, 2), Stimulus(3, 4)], 0)

a = a._as_speaker()
a.get_word(c)
print(a.language.lexicon)
print(a.language.lxc)

if c is None:
    print("discrimination failed")
else:
    print(c)