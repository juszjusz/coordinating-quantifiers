from objects.language import Language
from random import sample

class Population:

    def __init__(self, population_size: int):
        self.population_size = population_size
        self.population = [Agent(agent_id) for agent_id in range(population_size)]

    def select_pairs_per_round(self, games_per_round: int):
        agents_per_game = sample(self.population, games_per_round * 2)
        return [(agent1._as_hearer(), agent2._as_speaker()) for agent1, agent2 in zip(agents_per_game[::2], agents_per_game[1::2])]

class Agent:
    def __init__(self, id, language: Language = Language()):
        self.language = language
        self.id = id

    def _as_speaker(self):
        return SpeakerAgent(self.id, self.language)

    def _as_hearer(self):
        return HearerAgent(self.id, self.language)


class SpeakerAgent(Agent):
    def __init__(self, id, language: Language):
        Agent.__init__(self, id, language)

    def get_word(self, category):
        return self.language.pick_word(category)

    # getDiscriminativeCategory
    def get_discriminative_category(self, context: (int, int), topic):
        # TODO
        return None


class HearerAgent(Agent):
    def __init__(self, id, language: Language):
        Agent.__init__(self, id, language)

    def get_discriminative_category(self, word):
        return self.language.pick_category(word)

    def get_stimulus(self, category):
        # TODO
        return None
