from objects.language import Language
from objects.perception import Stimulus
from objects.perception import DiscriminativeCategory
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

    def discriminate(self, context, topic):
        s1 = context[0]
        s2 = context[1]
        responses1 = [c.response(s1) for c in self.language.discriminative_categories]
        responses2 = [c.response(s2) for c in self.language.discriminative_categories]
        max1 = max(responses1)
        max2 = max(responses2)
        max_args1 = [i for i, j in enumerate(responses1) if j == max1]
        max_args2 = [i for i, j in enumerate(responses2) if j == max2]
        i = max_args1[0]
        j = max_args2[0]
        return None if len(max_args1) != 1 or len(max_args2) != 1 or i != j else i if topic == 0 else j

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