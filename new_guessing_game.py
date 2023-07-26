from __future__ import division  # force python 3 division in python 2

import argparse
import logging
import os
import shutil
from dataclasses import dataclass
from random import randint
from typing import Dict, Tuple, Any, Callable

from agent import Speaker, Hearer, Agent, Population
from inmemory_calculus import load_inmemory_calculus, inmem

from stimulus import QuotientBasedStimulusFactory, NumericBasedStimulusFactory, ContextFactory


class GuessingGameAction:
    def __call__(self, speaker: Speaker, hearer: Hearer, context, topic, data_envelope: Dict) -> Any:
        pass


@dataclass
class GuessingGameStage:
    event_name: str
    event_action: GuessingGameAction

    @staticmethod
    def start_game():
        speaker_discrimination_game = DiscriminationGameAction(select_agent=select_speaker,
                                                               on_no_discrimination=GuessingGameStage.NO_DISCRIMINATION(),
                                                               on_no_noticeable_difference=GuessingGameStage.NO_NOTICEABLE_DIFFERENCE(),
                                                               on_no_category=GuessingGameStage.NO_CATEGORY(),
                                                               on_success=GuessingGameStage.DISCRIMINATION_GAME_SUCCESS())
        return GuessingGameStage("START", speaker_discrimination_game)

    @staticmethod
    def DISCRIMINATION_GAME_SUCCESS():
        return GuessingGameStage("DISCRIMINATION GAME SUCCESS", SpeakerPickMostConnectedWord(
            no_word_for_category=GuessingGameStage.NO_WORD_FOR_CATEGORY(),
            pick_most_connected_word=GuessingGameStage.PICK_MOST_CONNECTED_WORD_SUCCESS()))

    @staticmethod
    def PICK_MOST_CONNECTED_WORD_SUCCESS():
        return GuessingGameStage("PICK_MOST_CONNECTED_WORD", HearerGetMostConnectedHearerCategory(
            no_associated_categories=GuessingGameStage.NO_ASSOCIATED_CATEGORIES(),
            most_connected_category_success=GuessingGameStage.MOST_CONNECTED_HEARER_CATEGORY_SUCCESS()))

    @staticmethod
    def NO_CATEGORY():
        return GuessingGameStage("NO_CATEGORY", on_NO_DISCRIMINATION_ACTION(select_agent=select_hearer))

    @staticmethod
    def NO_NOTICEABLE_DIFFERENCE():
        return GuessingGameStage("NO_NOTICEABLE_DIFFERENCE", on_NO_NOTICEABLE_DIFFERENCE())

    @staticmethod
    def NO_DISCRIMINATION():
        return GuessingGameStage("NO_DISCRIMINATION", {})

    @staticmethod
    def NO_WORD_FOR_CATEGORY():
        return GuessingGameStage("NO_WORD_FOR_CATEGORY", on_NO_WORD_FOR_CATEGORY_SPEAKER())

    @staticmethod
    def NO_SUCH_WORD():
        hearer_discrimination_game = DiscriminationGameAction(select_agent=select_hearer,
                                                              on_no_discrimination=GuessingGameStage.NO_DISCRIMINATION(),
                                                              on_no_noticeable_difference=GuessingGameStage.NO_NOTICEABLE_DIFFERENCE(),
                                                              on_no_category=GuessingGameStage.NO_CATEGORY(),
                                                              on_success=GuessingGameStage.DISCRIMINATION_GAME_SUCCESS())

        return GuessingGameStage("NO_SUCH_WORD",
                                 on_NO_SUCH_WORD_HEARER_OR_NO_ASSOCIATED_CATEGORIES(hearer_discrimination_game))

    @staticmethod
    def NO_ASSOCIATED_CATEGORIES():
        return GuessingGameStage("NO_ASSOCIATED_CATEGORIES", {})

    @staticmethod
    def MOST_CONNECTED_HEARER_CATEGORY_SUCCESS():
        return GuessingGameStage("GET_MOST_CONNECTED_HEARER_CATEGORY SUCCESS", HearerGetTopicAction(
            hearer_get_topic_success=GuessingGameStage.HEARER_GET_TOPIC_SUCCESS()))

    @staticmethod
    def HEARER_GET_TOPIC_SUCCESS():
        return GuessingGameStage("HEARER_GET_TOPIC", OnComplete())

    @staticmethod
    def FAILURE():
        return GuessingGameStage("COMPLETE AND FAILURE", SuccessAction())

    @staticmethod
    def SUCCESS():
        return GuessingGameStage("COMPLETE AND FAILURE", FailureAction())


def select_speaker(speaker: Speaker, _: Hearer) -> Agent:
    return speaker


def select_hearer(_: Speaker, hearer: Hearer) -> Agent:
    return hearer


class DiscriminationGameAction(GuessingGameAction):

    def __init__(self, select_agent, on_no_category: GuessingGameStage,
                 on_no_noticeable_difference: GuessingGameStage, on_no_discrimination: GuessingGameStage,
                 on_success: GuessingGameStage):
        self.select_agent = select_agent
        self.on_no_category = on_no_category
        self.on_no_noticeable_difference = on_no_noticeable_difference
        self.on_no_discrimination = on_no_discrimination
        self.on_success = on_success

    def __call__(self, speaker: Speaker, hearer: Hearer, context, topic, data_envelope) -> GuessingGameStage:
        agent: Agent = self.select_agent(speaker, hearer)
        agent.store_ds_result(False)

        if not agent.categories:
            # raise NO_CATEGORY
            return self.on_no_category

        s1, s2 = context[0], context[1]

        if not s1.is_noticeably_different_from(s2):
            # raise NO_NOTICEABLE_DIFFERENCE
            return self.on_no_noticeable_difference

        i = agent.get_best_matching_category(s1)
        j = agent.get_best_matching_category(s2)

        if i == j:
            # raise NO_DISCRIMINATION
            return self.on_no_discrimination

        winning_category = agent.categories[i] if topic == 0 else agent.categories[j]

        winning_category.reinforce(context[topic], agent.beta)
        agent.forget_categories(winning_category)
        agent.switch_ds_result()

        winning_category_index = agent.categories.index(winning_category)
        data_envelope['winning_category_index'] = winning_category_index
        return self.on_success  # (winning_category=winning_category_index)


class SpeakerPickMostConnectedWord(GuessingGameAction):
    def __init__(self, no_word_for_category: GuessingGameStage, pick_most_connected_word: GuessingGameStage):
        self.no_word_for_category = no_word_for_category
        self.pick_most_connected_word = pick_most_connected_word

    def __call__(self, speaker: Speaker, hearer, context, topic, data_envelope):
        assert data_envelope['category'], 'category must be defined on this stage'
        category = data_envelope['category']

        # speaker_word = speaker.get_most_connected_word(category)

        # TEN WARUNEK NIE POWINIEN TU WYSTĄPIĆ, JEŚLI SIĘ POJAWIA,
        # OBSŁUŻYĆ NA WCZEŚNIEJSZYM ETAPIE
        # if category is None:
        #   raise ERROR

        if not speaker.lexicon or all(v == 0.0 for v in speaker.lxc.get_row_by_col(category)):
            # raise NO_WORD_FOR_CATEGORY
            return self.no_word_for_category
            # print("not words or all weights are zero")

        speaker_word = speaker.get_words_sorted_by_val(category)[0]
        logging.debug("Speaker(%d) says: %s" % (speaker.id, speaker_word))
        data_envelope['speaker_selected_word'] = self.speaker_word
        return self.pick_most_connected_word


class HearerGetMostConnectedHearerCategory(GuessingGameAction):

    def __init__(self, no_associated_categories: GuessingGameStage, most_connected_category_success: GuessingGameStage):
        self.no_associated_categories = no_associated_categories
        self.most_connected_category_success = most_connected_category_success

    def __call__(self, speaker, hearer, context, topic, data_envelope):
        word = data_envelope['speaker_word']

        # TEN WARUNEK NIE POWINIEN TU WYSTĄPIĆ, JEŚLI SIĘ POJAWIA,
        # OBSŁUŻYĆ NA WCZEŚNIEJSZYM ETAPIE
        # if word is None:
        #     raise ERROR

        if word not in hearer.lexicon:
            # raise NO_SUCH_WORD
            return GuessingGameStage.NO_SUCH_WORD()

        category_index, max_propensity = hearer.get_categories_sorted_by_val(word)[0]

        # TODO still happens
        if max_propensity == 0:
            logging.debug("\"%s\" has no associated categories" % word)
            # raise NO_ASSOCIATED_CATEGORIES
            return self.no_associated_categories

        # hearer_category = hearer.get_most_connected_category(speaker_word)
        data_envelope['category_index'] = category_index
        return self.most_connected_category_success


class HearerGetTopicAction(GuessingGameAction):
    def __init__(self, hearer_get_topic_success: GuessingGameStage):
        self.hearer_get_topic_success = hearer_get_topic_success

    def __call__(self, speaker: Speaker, hearer: Hearer, context, topic, **data_envelope) -> GuessingGameStage:
        assert 'category_index' in data_envelope
        category = data_envelope['category_index']
        # hearer_topic = hearer.get_topic(context=context, category=hearer_category)
        # TEN WARUNEK NIE POWINIEN TU WYSTĄPIĆ, JEŚLI SIĘ POJAWIA,
        # OBSŁUŻYĆ NA WCZEŚNIEJSZYM ETAPIE
        # if category is None:
        #     raise ERROR

        category = hearer.language.categories[category]
        topic = category.select(context)

        data_envelope['topic'] = topic
        return self.hearer_get_topic_success


class OnComplete(GuessingGameAction):

    def __call__(self, speaker, hearer, context, topic, **kwargs) -> GuessingGameStage:
        hearer_selected_topic = kwargs['topic']
        if hearer_selected_topic == topic:
            return GuessingGameStage.SUCCESS()
        else:
            return GuessingGameStage.FAILURE()


class SuccessAction(GuessingGameAction):

    def __call__(self, speaker, hearer, context, topic, **kwargs) -> GuessingGameStage:
        logging.debug("guessing game 1 success!")
        success1 = True
        speaker.store_cs1_result(success1)
        hearer.store_cs1_result(success1)
        # todo !!!
        # hearer.update_on_success(speaker_word, hearer_category)
        # speaker.update_on_success(speaker_word, speaker_category)
        return None


class FailureAction(GuessingGameAction):
    def __call__(self, speaker, hearer, context, topic, **kwargs) -> GuessingGameStage:
        success1 = False
        logging.debug("guessing game 1 failed!")
        speaker.store_cs1_result(success1)
        hearer.store_cs1_result(success1)
        # todo !!!
        # hearer.update_on_failure(speaker_word, hearer_category)
        # speaker.update_on_failure(speaker_word, speaker_category)
        return None


# class on_NO_CATEGORY_HEARER(GuessingGameAction):
#     def __call__(self, speaker, hearer, context, topic, **kwargs) -> GuessingGameStage:
#         # def on_NO_CATEGORY(self, agent, context, topic):
#         return on_NO_DISCRIMINATION_HEARER()(speaker, hearer, context, topic)
#
#
# class on_NO_CATEGORY_SPEAKER(GuessingGameAction):
#     def __call__(self, speaker, hearer, context, topic, **kwargs) -> GuessingGameStage:
#         # def on_NO_CATEGORY(self, agent, context, topic):
#         return on_NO_DISCRIMINATION_SPEAKER()(speaker, hearer, context, topic)


class on_NO_NOTICEABLE_DIFFERENCE(GuessingGameAction):
    def __call__(self, speaker, hearer, context, topic, **kwargs) -> GuessingGameStage:
        logging.debug("no noticeable difference")
        return None


class on_NO_DISCRIMINATION_ACTION(GuessingGameAction):
    def __init__(self, select_agent):
        self.select_agent = select_agent

    def __call__(self, speaker, hearer, context, topic, **kwargs) -> GuessingGameStage:
        agent = self.select_agent(speaker, hearer)
        logging.debug("no discrimination")
        logging.debug("%s(%d)" % (agent, agent.agent_id))
        agent.learn_stimulus(context, topic)
        return None


# class on_NO_DISCRIMINATION_HEARER(GuessingGameAction):
#     def __call__(self, speaker, hearer, context, topic, **kwargs) -> GuessingGameStage:
#         return on_NO_DISCRIMINATION_ACTION()(hearer, context, topic)
#
#
# class on_NO_DISCRIMINATION_SPEAKER(GuessingGameAction):
#     def __call__(self, speaker, hearer, context, topic, **kwargs) -> GuessingGameStage:
#         return on_NO_DISCRIMINATION_ACTION()(speaker, context, topic)

# to be move to Speaker subclass


class on_NO_WORD_FOR_CATEGORY_SPEAKER(GuessingGameAction):
    def __call__(self, speaker, hearer, context, topic, **kwargs) -> GuessingGameStage:
        agent_category = kwargs["agent_category"]
        logging.debug("%s(%d) has no word for his category" % (speaker, speaker.agent_id))
        new_word = speaker.add_new_word()
        # TODO speaker_word instead new_word_index?
        logging.debug("%s(%d) introduces new word \"%s\"" % (speaker, speaker.agent_id, new_word))
        speaker.learn_word_category(new_word, agent_category)
        logging.debug("%s(%d) associates \"%s\" with his category" % (speaker, speaker.agent_id, new_word))
        return None


class on_NO_SUCH_WORD_HEARER_OR_NO_ASSOCIATED_CATEGORIES(GuessingGameAction):
    def __init__(self, discriminate_game: DiscriminationGameAction):
        self.discriminate_game = discriminate_game

    def __call__(self, speaker, hearer, context, topic, **kwargs) -> GuessingGameStage:
        logging.debug('on no such word event Hearer(%d) adds word "{}"'.format(hearer, speaker_word))
        hearer.add_word(speaker_word)
        logging.debug("{} plays the discrimination game".format(hearer))
        # try:
        # category = hearer.discrimination_game(context, topic)
        next = self.discriminate_game(speaker, hearer, context, topic)
        logging.debug("Hearer(%d) category %d" % (hearer.id,
                                                  -1 if next is None else hearer.get_categories()[next].id))
        logging.debug("Hearer(%d) associates \"%s\" with his category %d" % (
            hearer.id, speaker_word, hearer.get_categories()[next].id))
        hearer.learn_word_category(speaker_word, next)

        return next
        # return category
        # except NO_CATEGORY:
        #     self.on_NO_CATEGORY(agent=hearer, context=context, topic=topic)
        #     TODO discuss
        # except NO_NOTICEABLE_DIFFERENCE:
        #     self.on_NO_NOTICEABLE_DIFFERENCE()
        # except NO_DISCRIMINATION:
        #     self.on_NO_DISCRIMINATION(agent=hearer, context=context, topic=topic)


class NewGuessingGame:

    def __init__(self, is_stage7_on=False):
        self.is_stage7_on = is_stage7_on
        # self.context = context
        # self.topic = 0

    # guessing game
    def play(self, speaker: Speaker, hearer: Hearer, context, topic):
        logging.debug("Stimulus 1: %s" % context[0])
        logging.debug("Stimulus 2: %s" % context[1])
        logging.debug("topic = %d" % (topic + 1))

        stage: GuessingGameStage = GuessingGameStage.start_game()
        data_envelope = {}
        while stage:
            logging.debug(stage.event_name)
            stage = stage.event_action(speaker, hearer, context, topic, data_envelope)

        # STAGE 7
        #
        # if self.is_stage7_on and self.completed and not success1:
        #     logging.debug("guessing game 2 starts!")
        #     word = None
        #     try:
        #         hearer_category2 = hearer.discrimination_game(self.context, self.topic)
        #         word, word_categories = hearer.select_word(category=hearer_category2)
        #     except NO_CATEGORY:
        #         self.exception_handler.on_NO_CATEGORY(agent=hearer, context=self.context, topic=self.topic)
        #     except NO_NOTICEABLE_DIFFERENCE:
        #         self.exception_handler.on_NO_NOTICEABLE_DIFFERENCE()
        #     except NO_DISCRIMINATION:
        #         self.exception_handler.on_NO_DISCRIMINATION(agent=hearer, context=self.context, topic=self.topic)
        #     except ERROR:
        #         self.exception_handler.on_LANGUAGE_ERROR()
        #
        #     success2 = word == discriminationGameEv
        #
        #     logging.debug("Hearer(%d) says %s" % (hearer.agent_id, word))
        #
        #     if success2:
        #         logging.debug("guessing game 2 success!")
        #         speaker.update_on_success_stage7(discriminationGameEv, speaker_category)
        #         hearer.update_on_success_stage7(word, word_categories)
        #     else:
        #         logging.debug("guessing game 2 failed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='quantifiers simulation')

    # parser.add_argument('--simulation_name', '-sn', help='simulation name', type=str, default='test')
    parser.add_argument('--population_size', '-p', help='population size', type=int, default=10)
    parser.add_argument('--stimulus', '-stm', help='quotient or numeric', type=str, default='quotient',
                        choices=['quotient', 'numeric'])
    parser.add_argument('--max_num', '-mn', help='max number for numerics or max denominator for quotients', type=int,
                        default=100)
    parser.add_argument('--discriminative_threshold', '-dt', help='discriminative threshold', type=float, default=.95)
    parser.add_argument('--delta_inc', '-dinc', help='delta increment', type=float, default=.2)
    parser.add_argument('--delta_dec', '-ddec', help='delta decrement', type=float, default=.2)
    parser.add_argument('--delta_inh', '-dinh', help='delta inhibition', type=float, default=.2)
    parser.add_argument('--alpha', '-a', help='forgetting rate', type=float, default=.01)
    parser.add_argument('--super_alpha', '-sa', help='complete forgetting of categories that have smaller weights',
                        type=float, default=.001)
    parser.add_argument('--beta', '-b', help='learning rate', type=float, default=0.2)
    parser.add_argument('--steps', '-s', help='number of steps', type=int, default=100)
    parser.add_argument('--runs', '-r', help='number of runs', type=int, default=1)
    parser.add_argument('--guessing_game_2', '-gg2', help='is the second stage of the guessing game on',
                        action='store_true')
    parser.add_argument('--in_mem_calculus_path', '-path', help='path to precomputed integrals', type=str,
                        default='inmemory_calculus')
    parser.add_argument('--seed', help='set seed value to replicate a random values', type=int, default=1)

    parsed_params = vars(parser.parse_args())

    log_levels = {'debug': logging.DEBUG, 'info': logging.INFO}
    # logging.basicConfig(stream=sys.stderr, level=log_levels[parsed_params['log_level']])
    load_inmemory_calculus(parsed_params['in_mem_calculus_path'], parsed_params['stimulus'])

    stimulus_factory = None
    if parsed_params['stimulus'] == 'quotient':
        stimulus_factory = QuotientBasedStimulusFactory(inmem['STIMULUS_LIST'], parsed_params['max_num'])
    if parsed_params['stimulus'] == 'numeric':
        stimulus_factory = NumericBasedStimulusFactory(inmem['STIMULUS_LIST'], parsed_params['max_num'])
    context_constructor = ContextFactory(stimulus_factory)

    population = Population(parsed_params, parsed_params['seed'])
    agents = [a for a in population]
    cxt = context_constructor()
    spaker = Speaker(agents[0])
    hearer = Hearer(agents[1])
    NewGuessingGame().play(spaker, hearer, cxt, 0)
    print(population)
