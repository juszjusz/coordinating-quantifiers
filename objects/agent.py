from objects.language import Language
from random import sample


class Population:

    def __init__(self, population_size: int):
        self.population_size = population_size
        self.population = [Agent(agent_id) for agent_id in range(population_size)]

    def select_pairs_per_round(self, games_per_round: int):
        agents_per_game = sample(self.population, games_per_round * 2)
        return [(a1._as_hearer(), a2._as_speaker()) for a1, a2 in zip(agents_per_game[::2], agents_per_game[1::2])]


class Agent:

    discriminative_threshold = 0.95

    def __init__(self, id, language: Language = Language()):
        self.language = language
        self.id = id
        self.discriminative_success = 0

    def discriminate(self, context, topic):
        if not self.language.categories:
            # print("no categories")
            self.language.add_category(context[topic])
            return None

        s1, s2 = context[0], context[1]
        responses1 = [c.response(s1) for c in self.language.categories]
        responses2 = [c.response(s2) for c in self.language.categories]
        # print(responses1)
        # print(responses2)
        max1 = max(responses1)
        max2 = max(responses2)
        max_args1 = [i for i, j in enumerate(responses1) if j == max1]
        max_args2 = [i for i, j in enumerate(responses2) if j == max2]
        # print(max_args1)
        # print(max_args2)

        if len(max_args1) > 1 or len(max_args2) > 1:
            raise Exception("Two categories give the same maximal value for stimulus")

        i = max_args1[0]
        j = max_args2[0]

        if i == j:
            if self.discriminative_success >= Agent.discriminative_threshold:
                self.language.update_category(i if topic == 0 else j, context[topic])
            else:
                self.language.add_category(context[topic])
            return None

        #discrimination successful
        return i if topic == 0 else j

    def _as_speaker(self):
        return SpeakerAgent(self.id, self.language)

    def _as_hearer(self):
        return HearerAgent(self.id, self.language)


class SpeakerAgent(Agent):
    def __init__(self, id, language: Language):
        Agent.__init__(self, id, language)

    def get_word(self, category):
        return self.language.get_word(category)

    # getDiscriminativeCategory
    def get_category(self, context: (int, int), topic):
        # TODO
        return None


class HearerAgent(Agent):
    def __init__(self, id, language: Language):
        Agent.__init__(self, id, language)

    def get_category(self, word):
        return self.language.pick_category(word)

    def get_stimulus(self, category):
        # TODO
        return None