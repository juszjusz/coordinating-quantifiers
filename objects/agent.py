from objects.language import Language
from random import sample
from enum import Enum


class Population:

    def __init__(self, population_size: int):d
        self.population_size = population_size
        self.agents = [Agent(agent_id) for agent_id in range(population_size)]

    def select_pairs_per_round(self, games_per_round: int):
        agents_per_game = sample(self.agents, games_per_round * 2)
        return [(a1, a2) for a1, a2 in zip(agents_per_game[::2], agents_per_game[1::2])]


class Agent(Language):

    class Role(Enum):
        SPEAKER = 1
        HEARER = 2

    discriminative_threshold = 0.95

    def __init__(self, id):
        Language.__init__(self)
        self.id = id
        self.discriminative_success = 0

    def discriminate(self, context, topic):
        if not self.categories:
            return None, Language.Error.NO_CATEGORY

        s1, s2 = context[0], context[1]
        responses1 = [c.response(s1) for c in self.categories]
        responses2 = [c.response(s2) for c in self.categories]
        max1, max2 = max(responses1), max(responses2)
        max_args1 = [i for i, j in enumerate(responses1) if j == max1]
        max_args2 = [i for i, j in enumerate(responses2) if j == max2]

        if len(max_args1) > 1 or len(max_args2) > 1:
            raise Exception("Two categories give the same maximal value for stimulus")

        i, j = max_args1[0], max_args2[0]

        if i == j:
            return None, Language.Error.NO_DISCRIMINATION

        #discrimination successful
        return i if topic == 0 else j, Agent.Error.NO_ERROR

    def get_stimulus(self, category):
        # TODO
        return None

    def learn_word_category(self, word, category_index):
        self.lxc[self.lexicon.index(word), category_index] = 0.5

    # def learn_word_topic(self, word: str, context: list, topic: int):
    #     c_j = self.discriminate(context, topic)
    #     w_i = len(self.lexicon)
    #     self.lexicon.append(word)
    #     rows_cnt, cols_cnt = self.lxc.shape
    #     self.lxc.resize((rows_cnt + 1, cols_cnt), refcheck=False)
    #     if c_j is None:
    #         # TODO question: is this ok or maybe learn_topic?
    #         return
    #     else:

    def learn_topic(self, category: int, context: list, topic: int):
        if self.discriminative_success >= Agent.discriminative_threshold and category is not None:
            self.update_category(category, context[topic])
        else:
            self.add_category(context[topic])

    def get_topic(self, context: list, category: int):
        if category is None:
            return None, Language.Error.ERROR

        category = self.categories[category]
        topic = category.select(context)
        return (topic, Language.Error.NO_DIFFERENCE) if topic is None \
            else (topic, Language.Error.NO_ERROR)

    def update(self, success: bool, role: Role, word, category):
        i = self.lexicon.index(word)
        c = category
        if success and role == self.Role.SPEAKER:
            self.lxc[i, c] = self.lxc[i, c] + 0.1*self.lxc[i, c]
            for k in range(len(self.categories)):
                if k != c:
                    self.lxc[i, k] = self.lxc[i, k] - 0.1 * self.lxc[i, k]

        elif success and role == self.Role.HEARER:
            self.lxc[i, c] = self.lxc[i, c] + 0.1 * self.lxc[i, c]
            for j in range(len(self.lexicon)):
                if j != i:
                    self.lxc[j, c] = self.lxc[j, c] - 0.1 * self.lxc[j, c]

        elif not success:
            self.lxc[i, c] = self.lxc[i, c] - 0.1 * self.lxc[i, c]

# class Speaker(Agent):
#     def __init__(self, agent):
#         super().__init__()
#         Agent.__init__(self, id)
#
#
# class Hearer(Agent):
#     def __init__(self, id):
#         Agent.__init__(self, id)
