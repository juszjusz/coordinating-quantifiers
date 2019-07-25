from __future__ import division  # force python 3 division in python 2
import logging
from agent import Agent
from guessing_game_exceptions import *
from perception import Perception
from perception import Stimulus
from random import choice


class GuessingGame:

    def __init__(self, speaker, hearer):
        self.completed = False
        self.context = [Stimulus(), Stimulus()]
        while not Perception.noticeable_difference(self.context[0], self.context[1]):
            self.context = [Stimulus(), Stimulus()]
        self.topic = choice([0, 1])
        self.exception_handler = ExceptionHandler()

    # guessing game
    def play(self, speaker, hearer):
        logging.debug("--")
        logging.debug(
            "Stimulus 1: %d/%d = %f" % (self.context[0].a, self.context[0].b, self.context[0].a / self.context[0].b))
        logging.debug(
            "Stimulus 2: %d/%d = %f" % (self.context[1].a, self.context[1].b, self.context[1].a / self.context[1].b))
        logging.debug("topic = %d" % (self.topic + 1))

        hearer_topic = None
        speaker_category = None
        speaker_word = None
        hearer_category = None

        try:
            speaker_category = speaker.discrimination_game(self.context, self.topic)
            speaker_word = speaker.get_word(speaker_category)
            hearer_category = hearer.get_category(word=speaker_word)
            hearer_topic = hearer.get_topic(context=self.context, category=hearer_category)
            self.completed = True
        except NO_CATEGORY:
            self.exception_handler.on_NO_CATEGORY(agent=speaker, context=self.context, topic=self.topic)
        except NO_NOTICEABLE_DIFFERENCE:
            self.exception_handler.on_NO_NOTICEABLE_DIFFERENCE()
        except NO_POSITIVE_RESPONSE_1:
            self.exception_handler.on_NO_POSITIVE_RESPONSE_1(agent=speaker, agent_category=speaker_category,
                                                             context=self.context)
        except NO_POSITIVE_RESPONSE_2:
            self.exception_handler.on_NO_POSITIVE_RESPONSE_2(agent=speaker, agent_category=speaker_category,
                                                             context=self.context)
        except NO_DISCRIMINATION_LOWER_1:
            self.exception_handler.on_NO_DISCRIMINATION_LOWER_1(agent=speaker,
                                                                agent_category=speaker_category,
                                                                context=self.context)
        except NO_DISCRIMINATION_LOWER_2:
            self.exception_handler.on_NO_DISCRIMINATION_LOWER_2(agent=speaker,
                                                                agent_category=speaker_category,
                                                                context=self.context)
        except NO_WORD_FOR_CATEGORY:
            self.exception_handler.on_NO_WORD_FOR_CATEGORY(agent=speaker, agent_category=speaker_category)
        except NO_ASSOCIATED_CATEGORIES:
            self.exception_handler.on_NO_ASSOCIATED_CATEGORIES(context=self.context,
                                                               topic=self.topic,
                                                               hearer=hearer,
                                                               speaker_word=speaker_word)
        except NO_DIFFERENCE_FOR_CATEGORY:
            hearer_category = self.exception_handler.on_NO_DIFFERENCE_FOR_CATEGORY(hearer=hearer,
                                                                                        context=self.context,
                                                                                        topic=self.topic,
                                                                                        speaker_word=speaker_word)
        except NO_SUCH_WORD:
            hearer_category = self.exception_handler.on_NO_SUCH_WORD(agent=hearer, context=self.context,
                                                                          topic=self.topic,
                                                                          speaker_word=speaker_word)
        except ERROR:
            self.exception_handler.on_LANGUAGE_ERROR()

        # logging.debug("discrimination success" if error == Agent.Error.NO_ERROR else "discrimination failure")

        success = self.topic == hearer_topic

        if self.completed and success:
            logging.debug("guessing game success!")
            speaker.store_cs_result(Agent.Result.SUCCESS)
            hearer.store_cs_result(Agent.Result.SUCCESS)
        else:
            logging.debug("guessing game failed!")
            speaker.store_cs_result(Agent.Result.FAILURE)
            hearer.store_cs_result(Agent.Result.FAILURE)

        if self.completed:
            speaker.update(success=success, role=Agent.Role.SPEAKER,
                                word=speaker_word, category=speaker_category)
            hearer.update(success=success, role=Agent.Role.HEARER,
                               word=speaker_word, category=hearer_category)

        # TODO stage 7
        # print('goto to stage 7')
        
        return self.completed and success

    def get_stats(self):
        return None


class ExceptionHandler:
    # to be move to Speaker Hearer subclass
    def on_NO_CATEGORY(self, agent, context, topic):
        logging.debug("no category")
        logging.debug("%s(%d)" % (agent, agent.id))
        agent.learn_stimulus(None, context, topic)

    # TODO wyrzucic?
    # to be move to Speaker Hearer subclass
    def on_NO_NOTICEABLE_DIFFERENCE(self):
        logging.debug("no noticeable difference")
        # to be move to Speaker Hearer subclass

    def on_NO_DISCRIMINATION_LOWER_1(self, agent, agent_category, context):
        logging.debug("no discrimination lower 1")
        logging.debug("%s(%d)" % (agent, agent.id))
        agent.learn_stimulus(agent_category, context, 0)

    # to be move to Speaker Hearer subclass
    def on_NO_DISCRIMINATION_LOWER_2(self, agent, agent_category, context):
        logging.debug("no discrimination lower 2")
        logging.debug("%s(%d)" % (agent, agent.id))
        agent.learn_stimulus(agent_category, context, 1)

    # to be move to Speaker Hearer subclass
    def on_NO_POSITIVE_RESPONSE_1(self, agent, agent_category, context):
        logging.debug("no responsive category for stimulus 1")
        logging.debug("%s(%d)" % (agent, agent.id))
        agent.learn_stimulus(agent_category, context, 0)

    # to be move to Speaker Hearer subclass
    def on_NO_POSITIVE_RESPONSE_2(self, agent, agent_category, context):
        logging.debug("no responsive category for stimulus 2")
        logging.debug("%s(%d)" % (agent, agent.id))
        agent.learn_stimulus(agent_category, context, 1)

    # to be move to Speaker and Hearer subclass
    def on_LANGUAGE_ERROR(self):
        pass

    # to be move to Speaker subclass
    def on_NO_WORD_FOR_CATEGORY(self, agent, agent_category):
        logging.debug("%s(%d) has no word for his category" % (agent, agent.id))
        new_word = agent.add_new_word()
        # TODO speaker_word instead new_word_index?
        logging.debug("%s(%d) introduces new word \"%s\"" % (agent, agent.id, new_word))
        agent.learn_word_category(new_word, agent_category)
        logging.debug("%s(%d) associates \"%s\" with his category" % (agent, agent.id, new_word))

    # to be move to Hearer subclass
    def on_NO_SUCH_WORD(self, agent, context, topic, speaker_word):
        logging.debug("Hearer(%d) adds word \"%s\"" % (agent.id, speaker_word))
        agent.add_word(speaker_word)
        logging.debug("Hearer(%d) plays the discrimination game" % agent.id)
        # TODO discrimination_game czy discriminate?
        category = None
        try:
            category = agent.discrimination_game(context, topic)
            logging.debug("Hearer(%d) category %d" % (agent.id,
                                                      -1 if category is None else category))
            logging.debug("Hearer(%d) associates \"%s\" with his category %d" % (
                agent.id, speaker_word, category))
            agent.learn_word_category(speaker_word, category)
            return category
        except NO_CATEGORY:
            self.on_NO_CATEGORY(agent=agent, context=context, topic=topic)
            # TODO discuss
        except NO_NOTICEABLE_DIFFERENCE:
            self.on_NO_NOTICEABLE_DIFFERENCE()
            # TODO do wywalenia?
            # raise Exception("Wow, there should be no error here.")
        except NO_POSITIVE_RESPONSE_1:
            self.on_NO_POSITIVE_RESPONSE_1(agent=agent, agent_category=None, context=context)
            # TODO discuss
            # raise Exception("Wow, there should be no error here.")
        except NO_POSITIVE_RESPONSE_2:
            self.on_NO_POSITIVE_RESPONSE_2(agent=agent, agent_category=None, context=context)
            # TODO discuss
            # raise Exception("Wow, there should be no error here.")
        except NO_DISCRIMINATION_LOWER_1:
            self.on_NO_DISCRIMINATION_LOWER_1(agent=agent, agent_category=None, context=context)
            # TODO discuss
            # raise Exception("Wow, there should be no error here.")
        except NO_DISCRIMINATION_LOWER_2:
            self.on_NO_DISCRIMINATION_LOWER_2(agent=agent, agent_category=None, context=context)
            # TODO discuss
            # raise Exception("Wow, there should be no error here.")

        logging.debug("Hearer(%d) category %-1")
        logging.debug("Hearer(%d) plays the discrimination game" % agent.id)

        try:
            category = agent.discrimination_game(context, topic)
            logging.debug("Hearer(%d) category %d" % (agent.id,
                                                      -1 if category is None else category))
            logging.debug("Hearer(%d) associates \"%s\" with his category %d" % (
                agent.id, speaker_word, category))
            agent.learn_word_category(speaker_word, category)
            return category
        except PerceptionError:
            logging.debug("Hearer(%d) - discrimination failure" % agent.id)
            logging.debug("Hearer(%d) does not associate \"%s\" with any category" % (agent.id, speaker_word))
            return None

    # HEArER
    def on_NO_DIFFERENCE_FOR_CATEGORY(self, hearer, context, topic, speaker_word):
        logging.debug("Hearer(%d) sees no difference between stimuli using his category" % (hearer.id))
        logging.debug("Hearer plays the discrimination game")
        # TODO discrimination_game czy discriminate?
        category = None
        try:
            category = hearer.discrimination_game(context, topic)
            logging.debug("Hearer(%d) associates \"%s\" with category %d" % (hearer.id, speaker_word, category + 1))
            hearer.learn_word_category(speaker_word, category)
            return category
        except NO_CATEGORY:
            self.on_NO_CATEGORY(agent=hearer, context=context, topic=topic)
            # raise Exception("Wow, there should be no error here.")
        except NO_NOTICEABLE_DIFFERENCE:
            self.on_NO_NOTICEABLE_DIFFERENCE()
            # raise Exception("Wow, there should be no error here.")
        except NO_POSITIVE_RESPONSE_1:
            self.on_NO_POSITIVE_RESPONSE_1(agent=hearer, agent_category=None, context=context)
            # raise Exception("Wow, there should be no error here.")
        except NO_POSITIVE_RESPONSE_2:
            self.on_NO_POSITIVE_RESPONSE_2(agent=hearer, agent_category=None, context=context)
            # raise Exception("Wow, there should be no error here.")
        except NO_DISCRIMINATION_LOWER_1:
            self.on_NO_DISCRIMINATION_LOWER_1(agent=hearer, agent_category=None, context=context)
            # raise Exception("Wow, there should be no error here.")
        except NO_DISCRIMINATION_LOWER_2:
            self.on_NO_DISCRIMINATION_LOWER_2(agent=hearer, agent_category=None, context=context)
            # raise Exception("Wow, there should be no error here.")

        try:
            logging.debug("Hearer plays the discrimination game")
            # TODO discrimination_game czy discriminate?
            category = hearer.discrimination_game(context, topic)
            logging.debug("Hearer(%d) associates \"%s\" with category %d" % (hearer.id, speaker_word, category + 1))
            hearer.learn_word_category(speaker_word, category)
            return category
        except PerceptionError:
            logging.debug("Hearer is unable to discriminate the topic")
            return None

    def on_NO_ASSOCIATED_CATEGORIES(self, hearer, context, topic, speaker_word):
        logging.debug("Hearer(%d) has not associate categories with %s" % (hearer.id, speaker_word))
        logging.debug("Hearer plays the discrimination game")
        # TODO discrimination_game czy discriminate?
        category = None
        try:
            category = hearer.discrimination_game(context, topic)
            logging.debug("Hearer(%d) associates \"%s\" with category %d" % (hearer.id, speaker_word, category + 1))
            hearer.learn_word_category(speaker_word, category)
            return category
        except NO_CATEGORY:
            self.on_NO_CATEGORY(agent=hearer, context=context, topic=topic)
            # raise Exception("Wow, there should be no error here.")
        except NO_NOTICEABLE_DIFFERENCE:
            self.on_NO_NOTICEABLE_DIFFERENCE()
            # raise Exception("Wow, there should be no error here.")
        except NO_POSITIVE_RESPONSE_1:
            self.on_NO_POSITIVE_RESPONSE_1(agent=hearer, agent_category=None, context=context)
            # raise Exception("Wow, there should be no error here.")
        except NO_POSITIVE_RESPONSE_2:
            self.on_NO_POSITIVE_RESPONSE_2(agent=hearer, agent_category=None, context=context)
            # raise Exception("Wow, there should be no error here.")
        except NO_DISCRIMINATION_LOWER_1:
            self.on_NO_DISCRIMINATION_LOWER_1(agent=hearer, agent_category=None, context=context)
            # raise Exception("Wow, there should be no error here.")
        except NO_DISCRIMINATION_LOWER_2:
            self.on_NO_DISCRIMINATION_LOWER_2(agent=hearer, agent_category=None, context=context)
            # raise Exception("Wow, there should be no error here.")

        logging.debug("Discrimination game failed")
        try:
            logging.debug("Hearer plays the discrimination game")
            # TODO discrimination_game czy discriminate?
            category = hearer.discrimination_game(context, topic)
            logging.debug("Hearer(%d) associates \"%s\" with category %d" % (hearer.id, speaker_word, category + 1))
            hearer.learn_word_category(speaker_word, category)
            return category
        except PerceptionError:
            logging.debug("Hearer is unable to discriminate the topic")
            return None
