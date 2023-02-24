from __future__ import division  # force python 3 division in python 2
import logging

from guessing_game_exceptions import *

from new_agent import NewSpeaker, NewHearer

states = {
    'on_start': 'ACTION',
    'complete': None
}


class GuessingGameEvent:
    def code(self) -> str:
        pass

    def __call__(self, speaker: NewSpeaker, hearer: NewHearer):
        pass


class DiscriminationGameWordFound(GuessingGameEvent):
    def __init__(self, word):
        self.word = word


class DiscriminationGameNoCategoryFound(GuessingGameEvent):

    def __call__(self, speaker: NewSpeaker, hearer: NewHearer):
        logging.debug("no category")
        hearer.learn_stimulus(context, topic)


class NewGuessingGame:

    def __init__(self, is_stage7_on, context):
        self.completed = False
        self.is_stage7_on = is_stage7_on
        self.context = context
        self.topic = 0
        self.exception_handler = ExceptionHandler()

    # guessing game
    def play(self, speaker: NewSpeaker, hearer: NewHearer):
        logging.debug("Stimulus 1: %s" % self.context[0])
        logging.debug("Stimulus 2: %s" % self.context[1])
        logging.debug("topic = %d" % (self.topic + 1))

        event: GuessingGameEvent = speaker.find_discriminating_word(self.context, self.topic)

        while event.code != 'complete':
            next_state = states[event.code()]
            event = next_state(speaker, hearer, event)

        try:
            event: GuessingGameEvent = speaker.find_discriminating_word(self.context, self.topic)
            while event.code != 'complete':
                next_state = states[event.code()]
                event = next_state(speaker, hearer, event)

            # raise NO_CATEGORY
            # raise NO_NOTICEABLE_DIFFERENCE
            # raise NO_DISCRIMINATION

            # logging.debug("Speaker(%d) says: %s" % (speaker.agent_id, discriminationGameEv))
            # hearer_topic = hearer.find_topic(self.context, discriminationGameEv)
            # logging.debug("Topic according to hearer(%d): %d" % (hearer.agent_id, hearer_topic + 1))
            # self.completed = True


        except NO_CATEGORY:
            self.exception_handler.on_NO_CATEGORY(agent=speaker, context=self.context, topic=self.topic)
        except NO_NOTICEABLE_DIFFERENCE:
            self.exception_handler.on_NO_NOTICEABLE_DIFFERENCE()
        except NO_DISCRIMINATION:
            self.exception_handler.on_NO_DISCRIMINATION(agent=speaker, context=self.context, topic=self.topic)
        except NO_WORD_FOR_CATEGORY:
            self.exception_handler.on_NO_WORD_FOR_CATEGORY(speaker=speaker, agent_category=speaker_category)
        except NO_ASSOCIATED_CATEGORIES:
            self.exception_handler.on_NO_ASSOCIATED_CATEGORIES(context=self.context,
                                                               topic=self.topic,
                                                               hearer=hearer,
                                                               speaker_word=discriminationGameEv)
        except NO_SUCH_WORD:
            self.exception_handler.on_NO_SUCH_WORD(hearer=hearer, context=self.context,
                                                   topic=self.topic, speaker_word=discriminationGameEv)
        except ERROR:
            self.exception_handler.on_LANGUAGE_ERROR()
        success1 = self.topic == hearer_topic

        if success1:
            logging.debug("guessing game 1 success!")
        else:
            logging.debug("guessing game 1 failed!")

        if self.completed and success1:
            speaker.update_on_success(discriminationGameEv, speaker_category)
            hearer.update_on_success(discriminationGameEv, hearer_category)
        elif self.completed:
            hearer.update_on_failure(discriminationGameEv, hearer_category)
            speaker.update_on_failure(discriminationGameEv, speaker_category)

        success2 = None
        # STAGE 7

        if self.is_stage7_on and self.completed and not success1:
            logging.debug("guessing game 2 starts!")
            word = None
            try:
                hearer_category2 = hearer.discrimination_game(self.context, self.topic)
                word, word_categories = hearer.select_word(category=hearer_category2)
            except NO_CATEGORY:
                self.exception_handler.on_NO_CATEGORY(agent=hearer, context=self.context, topic=self.topic)
            except NO_NOTICEABLE_DIFFERENCE:
                self.exception_handler.on_NO_NOTICEABLE_DIFFERENCE()
            except NO_DISCRIMINATION:
                self.exception_handler.on_NO_DISCRIMINATION(agent=hearer, context=self.context, topic=self.topic)
            except ERROR:
                self.exception_handler.on_LANGUAGE_ERROR()

            success2 = word == discriminationGameEv

            logging.debug("Hearer(%d) says %s" % (hearer.agent_id, word))

            if success2:
                logging.debug("guessing game 2 success!")
                speaker.update_on_success_stage7(discriminationGameEv, speaker_category)
                hearer.update_on_success_stage7(word, word_categories)
            else:
                logging.debug("guessing game 2 failed!")


class ExceptionHandler:
    # to be move to Speaker Hearer subclass
    def on_NO_CATEGORY(self, agent, context, topic):
        logging.debug("no category")
        logging.debug("%s(%d)" % (agent, agent.agent_id))
        agent.learn_stimulus(context, topic)

    # TODO wyrzucic?
    # to be move to Speaker Hearer subclass
    def on_NO_NOTICEABLE_DIFFERENCE(self):
        logging.debug("no noticeable difference")
        # to be move to Speaker Hearer subclass

    def on_NO_DISCRIMINATION(self, agent, context, topic):
        logging.debug("no discrimination")
        logging.debug("%s(%d)" % (agent, agent.agent_id))
        agent.learn_stimulus(context, topic)

    # to be move to Speaker and Hearer subclass
    def on_LANGUAGE_ERROR(self):
        logging.debug("Unknown error")
        exit()

    # to be move to Speaker subclass
    def on_NO_WORD_FOR_CATEGORY(self, speaker, agent_category):
        logging.debug("%s(%d) has no word for his category" % (speaker, speaker.agent_id))
        new_word = speaker.add_new_word()
        # TODO speaker_word instead new_word_index?
        logging.debug("%s(%d) introduces new word \"%s\"" % (speaker, speaker.agent_id, new_word))
        speaker.learn_word_category(new_word, agent_category)
        logging.debug("%s(%d) associates \"%s\" with his category" % (speaker, speaker.agent_id, new_word))

    # to be move to Hearer subclass
    def on_NO_SUCH_WORD(self, hearer, context, topic, speaker_word):
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
