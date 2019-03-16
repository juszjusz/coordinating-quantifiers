from objects.nature import Nature
from objects.player import PlayersPair, Player
from random import choice

def play_game(player1: Player, player2: Player, nature: Nature, rounds: int):
    players_pair = PlayersPair(player1, player2)
    for round in range(0, rounds):
        print('playing round', round)
        play_round(players_pair, nature)

def play_round(players_pair: PlayersPair, nature: Nature):
    speaker, hearer = players_pair.pick_speaker_and_hearer()
    context = nature.emit_context()
    topic = choice(context)

    speaker_category = speaker.pick_category(context, topic)
    speaker_word = speaker.pick_word(speaker_category)

    hearer_category = hearer.pick_category(speaker_word)
    hearer_stimulus = hearer.pick_stimulus(hearer_category)

    if topic == hearer_stimulus:
        print('game finished')

    else:
        print('goto to stage 7')