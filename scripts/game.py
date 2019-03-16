from scripts.game_definitions import play_game
from objects.player import Player
from objects.nature import Nature
from objects.language import Language

player1 = Player('player1', Language({'blue': ['BLUE', 'BLUEISH', 'VIOLET'], 'red': ['RED'], 'blueish': ['BLUEISH']}))
player2 = Player('player2', Language({'blue': ['BLUE', 'VIOLET', 'PURPLE'], 'red': ['ORANGE'], 'blueish': ['BLUEISH']}))
nature = Nature()
rounds = 10

play_game(player1, player2, nature, rounds)