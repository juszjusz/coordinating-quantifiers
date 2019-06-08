from scripts.game_definitions import play_game
from objects.agent import Agent
from objects.nature import Nature
from objects.language import Language

player1 = Agent('player1', Language())
player2 = Agent('player2', Language())
nature = Nature()
rounds = 10

play_game(player1, player2, nature, rounds)