from objects.nature import Nature
from objects.player import PlayersPair, Player
from random import choice

def play_game(player1: Player, player2: Player, nature: Nature, rounds: int):
    players_pair = PlayersPair(player1, player2)
    for round in range(0, rounds):
        print('playing round', round)
        play_round(players_pair, nature)

# guessing game
def play_round(players_pair: PlayersPair, nature: Nature):
    speaker, hearer = players_pair.get_speaker_and_hearer()
    context = nature.emit_context()
    topic = choice(context)

    # 2. the speaker tries to discriminate the topic from the context by playing the discrimination game
    speaker_category = speaker.get_discriminative_category(context, topic)
    # 3. the speaker looks up the word forms in Ds associated with speaker_category.
    speaker_word = speaker.get_word(speaker_category)
    # 4. the hearer looks up speaker_word in his lexicon... (happens 'under the hood')
    # 5. the hearer does have speaker_word in his lexicon...
    hearer_category = hearer.get_discriminative_category(speaker_word)
    # 5. point to the stimulus ...
    hearer_stimulus = hearer.get_stimulus(hearer_category)
    # 6. The speaker observes to which stimulus the hearer is pointing and if
    if topic == hearer_stimulus:
        print('game finished')
        speaker
    # otherwise
    else:
        print('goto to stage 7')

# my comments:
# 1. Lets make method naming consistent, i.e. if a method is used for queries, then
# name it 'get_' + SUFFIX (or 'select_' + SUFFIX, but let it be standarized)
# 2. Follow python' naming convention (underscres instead of camelcase, i.e. 'get_word' instead of 'getWord')
# 3. Make naming agnostic with respect to implementation ('updateDictionary' instead of 'updateAssocationMatrix')