from __future__ import division  # force python 3 division in python 2
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Any

from guessing_game_exceptions import *

from new_agent import NewSpeaker, NewHearer


@dataclass
class EventResult:
    complete: bool
    event_name: str
    emitted_values: Dict[str, Any]

    @staticmethod
    def start_event():
        return EventResult(False, "start", {})

    @staticmethod
    def discrimination_game_event():
        return EventResult(False, "discrimination game", {})

    @staticmethod
    def NO_CATEGORY():
        return EventResult(True, "NO_CATEGORY", {})

    @staticmethod
    def NO_NOTICEABLE_DIFFERENCE():
        return EventResult(True, "NO_NOTICEABLE_DIFFERENCE", {})

    @staticmethod
    def NO_DISCRIMINATION():
        return EventResult(True, "NO_DISCRIMINATION", {})

    @staticmethod
    def NO_WORD_FOR_CATEGORY():
        return EventResult(False, "NO_WORD_FOR_CATEGORY", {})

    @staticmethod
    def DISCRIMINATION_GAME_SUCCESS(**kwargs):
        return EventResult(False, "DISCRIMINATION GAME SUCCESS", kwargs)

    @staticmethod
    def PICK_MOST_CONNECTED_WORD(**kwargs):
        return EventResult(False, "PICK_MOST_CONNECTED_WORD", kwargs)

    @staticmethod
    def NO_SUCH_WORD():
        return EventResult(True, "NO_SUCH_WORD", {})

    @staticmethod
    def NO_ASSOCIATED_CATEGORIES():
        return EventResult(True, "NO_ASSOCIATED_CATEGORIES", {})

    @staticmethod
    def GET_MOST_CONNECTED_HEARER_CATEGORY(category_index):
        return EventResult(False, "GET_MOST_CONNECTED_HEARER_CATEGORY", {"category_index": category_index})

    @staticmethod
    def HEARER_GET_TOPIC(topic):
        return EventResult(False, "HEARER_GET_TOPIC", {"topic": topic})

    @staticmethod
    def FAILURE():
        return EventResult(True, "COMPLETE AND FAILURE", {})

    @staticmethod
    def SUCCESS():
        return EventResult(True, "COMPLETE AND FAILURE", {})


class GuessingGameAction:
    def __call__(self, speaker, hearer, context, topic, **kwargs) -> EventResult:
        pass


class StartGameAction(GuessingGameAction):

    def __call__(self, speaker, hearer, context, topic, **kwargs) -> EventResult:
        return EventResult.start_event()


class SpeakerDiscriminationGameAction(GuessingGameAction):

    def __call__(self, speaker, hearer, context, topic, **kwargs) -> EventResult:
        speaker.store_ds_result(False)

        if not speaker.categories:
            # raise NO_CATEGORY
            return EventResult.NO_CATEGORY()

        s1, s2 = context[0], context[1]

        if not s1.is_noticeably_different_from(s2):
            # raise NO_NOTICEABLE_DIFFERENCE
            return EventResult.NO_NOTICEABLE_DIFFERENCE()

        i = speaker.get_best_matching_category(s1)
        j = speaker.get_best_matching_category(s2)

        if i == j:
            # raise NO_DISCRIMINATION
            return EventResult.NO_DISCRIMINATION()

        winning_category = speaker.categories[i] if topic == 0 else speaker.categories[j]

        winning_category.reinforce(context[topic], speaker.beta)
        speaker.forget_categories(winning_category)
        speaker.switch_ds_result()

        winning_category_index = speaker.categories.index(winning_category)
        return EventResult.DISCRIMINATION_GAME_SUCCESS(winning_category=winning_category_index)


class SpeakerPickMostConnectedWord(GuessingGameAction):
    def __call__(self, speaker, hearer, context, topic, **kwargs):
        category = kwargs['speaker_category']
        # speaker_word = speaker.get_most_connected_word(category)

        # TEN WARUNEK NIE POWINIEN TU WYSTĄPIĆ, JEŚLI SIĘ POJAWIA,
        # OBSŁUŻYĆ NA WCZEŚNIEJSZYM ETAPIE
        # if category is None:
        #   raise ERROR

        if not speaker.lexicon or all(v == 0.0 for v in speaker.lxc.get_row_by_col(category)):
            # raise NO_WORD_FOR_CATEGORY
            return EventResult.NO_WORD_FOR_CATEGORY()
            # print("not words or all weights are zero")

        speaker_word = speaker.get_words_sorted_by_val(category)[0]
        logging.debug("Speaker(%d) says: %s" % (speaker.id, speaker_word))
        return EventResult.PICK_MOST_CONNECTED_WORD(speaker_word=speaker_word)


class HearerGetMostConnectedHearerCategory(GuessingGameAction):
    def __call__(self, speaker, hearer, context, topic, **kwargs):
        word = kwargs['speaker_word']

        # TEN WARUNEK NIE POWINIEN TU WYSTĄPIĆ, JEŚLI SIĘ POJAWIA,
        # OBSŁUŻYĆ NA WCZEŚNIEJSZYM ETAPIE
        # if word is None:
        #     raise ERROR

        if word not in hearer.lexicon:
            # raise NO_SUCH_WORD
            return EventResult.NO_SUCH_WORD()

        category_index, max_propensity = hearer.get_categories_sorted_by_val(word)[0]

        # TODO still happens
        if max_propensity == 0:
            logging.debug("\"%s\" has no associated categories" % word)
            # raise NO_ASSOCIATED_CATEGORIES
            return EventResult.NO_ASSOCIATED_CATEGORIES()

        # hearer_category = hearer.get_most_connected_category(speaker_word)
        return EventResult.GET_MOST_CONNECTED_HEARER_CATEGORY(category_index=category_index)


class HearerGetTopic(GuessingGameAction):

    def __call__(self, speaker, hearer, context, topic, **kwargs) -> EventResult:
        category = kwargs["category_index"]
        # hearer_topic = hearer.get_topic(context=context, category=hearer_category)
        # TEN WARUNEK NIE POWINIEN TU WYSTĄPIĆ, JEŚLI SIĘ POJAWIA,
        # OBSŁUŻYĆ NA WCZEŚNIEJSZYM ETAPIE
        # if category is None:
        #     raise ERROR

        category = hearer.language.categories[category]
        topic = category.select(context)
        return EventResult.HEARER_GET_TOPIC(topic=topic)


class OnComplete(GuessingGameAction):

    def __call__(self, speaker, hearer, context, topic, **kwargs) -> EventResult:
        hearer_selected_topic = kwargs['topic']
        if hearer_selected_topic == topic:
            return EventResult.SUCCESS()
        else:
            return EventResult.FAILURE()


class Success(GuessingGameAction):

    def __call__(self, speaker, hearer, context, topic, **kwargs) -> EventResult:
        logging.debug("guessing game 1 success!")
        success1 = True
        speaker.store_cs1_result(success1)
        hearer.store_cs1_result(success1)
        hearer.update_on_success(speaker_word, hearer_category)
        speaker.update_on_success(speaker_word, speaker_category)
        return None


class Failure(GuessingGameAction):
    def __call__(self, speaker, hearer, context, topic, **kwargs) -> EventResult:
        success1 = False
        logging.debug("guessing game 1 failed!")
        speaker.store_cs1_result(success1)
        hearer.store_cs1_result(success1)
        hearer.update_on_failure(speaker_word, hearer_category)
        speaker.update_on_failure(speaker_word, speaker_category)
        return None


class on_NO_CATEGORY_HEARER(GuessingGameAction):
    def __call__(self, speaker, hearer, context, topic, **kwargs) -> EventResult:
        # def on_NO_CATEGORY(self, agent, context, topic):
        logging.debug("no category")
        logging.debug("%s(%d)" % (hearer, hearer.agent_id))
        hearer.learn_stimulus(context, topic)
        return None


class on_NO_CATEGORY_SPEAKER(GuessingGameAction):
    def __call__(self, speaker, hearer, context, topic, **kwargs) -> EventResult:
        # def on_NO_CATEGORY(self, agent, context, topic):
        logging.debug("no category")
        logging.debug("%s(%d)" % (speaker, speaker.agent_id))
        speaker.learn_stimulus(context, topic)
        return None


class on_NO_NOTICEABLE_DIFFERENCE(GuessingGameAction):
    def __call__(self, speaker, hearer, context, topic, **kwargs) -> EventResult:
        logging.debug("no noticeable difference")
        return None


class on_NO_DISCRIMINATION_HEARER(GuessingGameAction):
    def __call__(self, speaker, hearer, context, topic, **kwargs) -> EventResult:
        logging.debug("no discrimination")
        logging.debug("%s(%d)" % (hearer, hearer.agent_id))
        hearer.learn_stimulus(context, topic)
        return None


class on_NO_DISCRIMINATION_SPEAKER(GuessingGameAction):
    def __call__(self, speaker, hearer, context, topic, **kwargs) -> EventResult:
        logging.debug("no discrimination")
        logging.debug("%s(%d)" % (speaker, speaker.agent_id))
        speaker.learn_stimulus(context, topic)
        return None

    # to be move to Speaker subclass


class on_NO_WORD_FOR_CATEGORY:
    def __call__(self, speaker, hearer, context, topic, **kwargs) -> EventResult:
        agent_category = kwargs["agent_category"]
        logging.debug("%s(%d) has no word for his category" % (speaker, speaker.agent_id))
        new_word = speaker.add_new_word()
        # TODO speaker_word instead new_word_index?
        logging.debug("%s(%d) introduces new word \"%s\"" % (speaker, speaker.agent_id, new_word))
        speaker.learn_word_category(new_word, agent_category)
        logging.debug("%s(%d) associates \"%s\" with his category" % (speaker, speaker.agent_id, new_word))
        return None


class on_NO_SUCH_WORD:

    def __call__(self, speaker, hearer, context, topic, **kwargs) -> EventResult:
        speaker_word = None
        logging.debug("Hearer(%d) adds word \"%s\"" % (hearer.agent_id, speaker_word))
        hearer.add_word(speaker_word)
        logging.debug("Hearer(%d) plays the discrimination game" % hearer.agent_id)
        # TODO discrimination_game czy discriminate?
        category = None
        try:
            category = hearer.find_discriminating_word(context, topic)
            logging.debug("Hearer(%d) category %d" % (hearer.agent_id,
                                                      -1 if category is None else hearer.get_categories()[
                                                          category].agent_id))
            logging.debug("Hearer(%d) associates \"%s\" with his category %d" % (
                hearer.agent_id, speaker_word, hearer.get_categories()[category].agent_id))
            hearer.learn_word_category(speaker_word, category)
            # return category
        except NO_CATEGORY:
            self.on_NO_CATEGORY(agent=hearer, context=context, topic=topic)
            # TODO discuss
        except NO_NOTICEABLE_DIFFERENCE:
            self.on_NO_NOTICEABLE_DIFFERENCE()
        except NO_DISCRIMINATION:
            self.on_NO_DISCRIMINATION(agent=hearer, context=context, topic=topic)

    def on_NO_ASSOCIATED_CATEGORIES(self, hearer, context, topic, speaker_word):
        logging.debug("Hearer(%d) has not associate categories with %s" % (hearer.agent_id, speaker_word))
        logging.debug("Hearer plays the discrimination game")
        # TODO discrimination_game czy discriminate?
        category = None
        try:
            category = hearer.find_discriminating_word(context, topic)
            logging.debug("Hearer(%d) associates \"%s\" with category %d" % (
                hearer.agent_id, speaker_word, hearer.get_categories()[category].agent_id))
            hearer.learn_word_category(speaker_word, category)
        except NO_CATEGORY:
            self.on_NO_CATEGORY(agent=hearer, context=context, topic=topic)
        except NO_NOTICEABLE_DIFFERENCE:
            self.on_NO_NOTICEABLE_DIFFERENCE()
        except NO_DISCRIMINATION:
            self.on_NO_DISCRIMINATION(agent=hearer, context=context, topic=topic, category_index=category)


class NewGuessingGame:

    def __init__(self, is_stage7_on, context):
        self.is_stage7_on = is_stage7_on
        self.context = context
        self.topic = 0

    # guessing game
    def play(self, speaker: NewSpeaker, hearer: NewHearer):
        logging.debug("Stimulus 1: %s" % self.context[0])
        logging.debug("Stimulus 2: %s" % self.context[1])
        logging.debug("topic = %d" % (self.topic + 1))

        actions: Dict[str, GuessingGameAction] = {
            "start": StartGameAction(),
            "discrimination game": SpeakerDiscriminationGameAction()
        }

        action = actions["start"]
        event: EventResult = action(speaker, hearer, self.context, self.topic)
        while not event.complete:
            action = actions[event.event_name]
            event = action(speaker, hearer, self.context, self.topic, **event.emitted_values)

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
