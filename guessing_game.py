from __future__ import division  # force python 3 division in python 2
import logging
from agent import Agent
from guessing_game_exceptions import *
from perception import Perception
from perception import Stimulus
from random import choice


class GuessingGame:

    def __init__(self, speaker, hearer):
        self.speaker = speaker
        self.hearer = hearer
        self.completed = False
        self.context = [Stimulus(), Stimulus()]
        while not Perception.noticeable_difference(self.context[0], self.context[1]):
            self.context = [Stimulus(), Stimulus()]
        self.topic = choice([0, 1])
        self.hearer_topic = None
        self.speaker_category = None
        self.speaker_word = None
        self.hearer_category = None
        self.hearer_word = None
        self.exception_handler = ExceptionHandler()

    # guessing game
    def play(self):
        logging.debug("--")
        logging.debug(
            "Stimulus 1: %d/%d = %f" % (self.context[0].a, self.context[0].b, self.context[0].a / self.context[0].b))
        logging.debug(
            "Stimulus 2: %d/%d = %f" % (self.context[1].a, self.context[1].b, self.context[1].a / self.context[1].b))
        logging.debug("topic = %d" % (self.topic + 1))

        try:
            self.speaker_category = self.speaker.discrimination_game(self.context, self.topic)
            self.speaker.store_ds_result(self.speaker.Result.SUCCESS)
            self.speaker_word = self.speaker.get_word(self.speaker_category)
            self.hearer_category = self.hearer.get_category(word=self.speaker_word)
            self.hearer_topic = self.hearer.get_topic(context=self.context, category=self.hearer_category)
            self.completed = True
        except NO_CATEGORY:
            self.speaker.store_ds_result(self.speaker.Result.FAILURE)
            self.exception_handler.on_NO_CATEGORY(agent=self.speaker, context=self.context, topic=self.topic)
        except NO_NOTICEABLE_DIFFERENCE:
            self.speaker.store_ds_result(self.speaker.Result.FAILURE)
            self.exception_handler.on_NO_NOTICEABLE_DIFFERENCE()
        except NO_POSITIVE_RESPONSE_1:
            self.speaker.store_ds_result(self.speaker.Result.FAILURE)
            self.exception_handler.on_NO_POSITIVE_RESPONSE_1(agent=self.speaker, agent_category=self.speaker_category,
                                                             context=self.context)
        except NO_POSITIVE_RESPONSE_2:
            self.speaker.store_ds_result(self.speaker.Result.FAILURE)
            self.exception_handler.on_NO_POSITIVE_RESPONSE_2(agent=self.speaker, agent_category=self.speaker_category,
                                                             context=self.context)
        except NO_DISCRIMINATION_LOWER_1:
            self.speaker.store_ds_result(self.speaker.Result.FAILURE)
            self.exception_handler.on_NO_DISCRIMINATION_LOWER_1(agent=self.speaker,
                                                                agent_category=self.speaker_category,
                                                                context=self.context)
        except NO_DISCRIMINATION_LOWER_2:
            self.speaker.store_ds_result(self.speaker.Result.FAILURE)
            self.exception_handler.on_NO_DISCRIMINATION_LOWER_2(agent=self.speaker,
                                                                agent_category=self.speaker_category,
                                                                context=self.context)
        except NO_WORD_FOR_CATEGORY:
            self.exception_handler.on_NO_WORD_FOR_CATEGORY(agent=self.speaker, agent_category=self.speaker_category)
        except NO_DIFFERENCE_FOR_CATEGORY:
            self.hearer_category = self.exception_handler.on_NO_DIFFERENCE_FOR_CATEGORY(agent=self.hearer,
                                                                                        context=self.context,
                                                                                        topic=self.topic,
                                                                                        speaker_word=self.speaker_word)
        except NO_SUCH_WORD:
            self.hearer_category = self.exception_handler.on_NO_SUCH_WORD(agent=self.hearer, context=self.context,
                                                                          topic=self.topic,
                                                                          speaker_word=self.speaker_word)
        except ERROR:
            self.exception_handler.on_LANGUAGE_ERROR()

        # logging.debug("discrimination success" if error == Agent.Error.NO_ERROR else "discrimination failure")

        success = self.topic == self.hearer_topic

        if self.completed and success:
            logging.debug("guessing game success!")
            self.speaker.store_cs_result(Agent.Result.SUCCESS)
            self.hearer.store_cs_result(Agent.Result.SUCCESS)
        else:
            logging.debug("guessing game failed!")
            self.speaker.store_cs_result(Agent.Result.FAILURE)
            self.hearer.store_cs_result(Agent.Result.FAILURE)

        if self.completed:
            self.speaker.update(success=success, role=Agent.Role.SPEAKER,
                                word=self.speaker_word, category=self.speaker_category)
            self.hearer.update(success=success, role=Agent.Role.HEARER,
                               word=self.speaker_word, category=self.hearer_category)

        if self.speaker.id == 0:
            logging.debug("Agent(0) language")
            logging.debug(self.speaker.lexicon)
            logging.debug(self.speaker.lxc)
            logging.debug("Agent(1) language")
            logging.debug(self.hearer.lexicon)
            logging.debug(self.hearer.lxc)
        else:
            logging.debug("Agent(0) language")
            logging.debug(self.hearer.lexicon)
            logging.debug(self.hearer.lxc)
            logging.debug("Agent(1) language")
            logging.debug(self.speaker.lexicon)
            logging.debug(self.speaker.lxc)

        # TODO stage 7
        # print('goto to stage 7')
        return self.completed and success

    def get_stats(self):
        return None


class ExceptionHandler:
    # to be move to Speaker Hearer subclass
    def on_NO_CATEGORY(self, agent: Agent, context, topic):
        logging.debug("no category")
        logging.debug("%s(%d)" % (agent, agent.id))
        agent.learn_stimulus(None, context, topic)

    # to be move to Speaker Hearer subclass
    def on_NO_NOTICEABLE_DIFFERENCE(self):
        logging.debug("no noticeable difference")
        # to be move to Speaker Hearer subclass

    def on_NO_DISCRIMINATION_LOWER_1(self, agent: Agent, agent_category, context):
        logging.debug("no discrimination lower 1")
        logging.debug("%s(%d)" % (agent, agent.id))
        agent.learn_stimulus(agent_category, context, 0)

    # to be move to Speaker Hearer subclass
    def on_NO_DISCRIMINATION_LOWER_2(self, agent: Agent, agent_category, context):
        logging.debug("no discrimination lower 2")
        logging.debug("%s(%d)" % (agent, agent.id))
        agent.learn_stimulus(agent_category, context, 1)

    # to be move to Speaker Hearer subclass
    def on_NO_POSITIVE_RESPONSE_1(self, agent: Agent, agent_category, context):
        logging.debug("no responsive category for stimulus 1")
        logging.debug("%s(%d)" % (agent, agent.id))
        agent.learn_stimulus(agent_category, context, 0)

    # to be move to Speaker Hearer subclass
    def on_NO_POSITIVE_RESPONSE_2(self, agent: Agent, agent_category, context):
        logging.debug("no responsive category for stimulus 2")
        logging.debug("%s(%d)" % (agent, agent.id))
        agent.learn_stimulus(agent_category, context, 1)

    # to be move to Speaker and Hearer subclass
    def on_LANGUAGE_ERROR(self):
        pass

    # to be move to Speaker subclass
    def on_NO_WORD_FOR_CATEGORY(self, agent: Agent, agent_category):
        logging.debug("%s(%d) has no word for his category" % (agent, agent.id))
        new_word = agent.add_new_word()
        # TODO speaker_word instead new_word_index?
        logging.debug("%s(%d) introduces new word \"%s\"" % (agent, agent.id, new_word))
        agent.learn_word_category(new_word, agent_category)
        logging.debug("%s(%d) associates \"%s\" with his category" % (agent, agent.id, new_word))
        return

    # to be move to Hearer subclass
    def on_NO_SUCH_WORD(self, agent: Agent, context, topic, speaker_word):
        # TODO added recursion, invocation of on_error, check
        logging.debug("Hearer(%d) adds word \"%s\"" % (agent.id, speaker_word))
        agent.add_word(speaker_word)
        logging.debug("Hearer(%d) plays the discrimination game" % agent.id)
        # TODO discrimination_game czy discriminate?
        try:
            category = agent.discrimination_game(context, topic)
            logging.debug("Hearer(%d) category %d" % (agent.id,
                                                      -1 if category is None else category))
        except NO_CATEGORY:
            self.on_NO_CATEGORY(agent=agent, context=context, topic=topic)
            logging.debug("Hearer(%d) plays the discrimination game" % agent.id)
            category = agent.discrimination_game(context, topic)
            # TODO discuss
            # raise Exception("Wow, there should be no error here.")
        except NO_NOTICEABLE_DIFFERENCE:
            self.on_NO_NOTICEABLE_DIFFERENCE()
            logging.debug("Hearer(%d) plays the discrimination game" % agent.id)
            category = agent.discrimination_game(context, topic)
            # TODO discuss
            # raise Exception("Wow, there should be no error here.")
        except NO_POSITIVE_RESPONSE_1:
            self.on_NO_POSITIVE_RESPONSE_1(agent=agent, agent_category=None, context=context)
            logging.debug("Hearer(%d) plays the discrimination game" % agent.id)
            category = agent.discrimination_game(context, topic)
            # TODO discuss
            # raise Exception("Wow, there should be no error here.")
        except NO_POSITIVE_RESPONSE_2:
            self.on_NO_POSITIVE_RESPONSE_2(agent=agent, agent_category=None, context=context)
            logging.debug("Hearer(%d) plays the discrimination game" % agent.id)
            category = agent.discrimination_game(context, topic)
            # TODO discuss
            # raise Exception("Wow, there should be no error here.")
        except NO_DISCRIMINATION_LOWER_1:
            self.on_NO_DISCRIMINATION_LOWER_1(agent=agent, agent_category=None, context=context)
            logging.debug("Hearer(%d) plays the discrimination game" % agent.id)
            category = agent.discrimination_game(context, topic)
            # TODO discuss
            # raise Exception("Wow, there should be no error here.")
        except NO_DISCRIMINATION_LOWER_2:
            self.on_NO_DISCRIMINATION_LOWER_2(agent=agent, agent_category=None, context=context)
            logging.debug("Hearer(%d) plays the discrimination game" % agent.id)
            category = agent.discrimination_game(context, topic)
            # TODO discuss
            # raise Exception("Wow, there should be no error here.")

        logging.debug("Hearer(%d) associates \"%s\" with his category %d" % (
            agent.id, speaker_word, category))
        agent.learn_word_category(speaker_word, category)
        return category

    # HEArER
    def on_NO_DIFFERENCE_FOR_CATEGORY(self, agent: Agent, context, topic, speaker_word):
        logging.debug("%s(%d) sees no difference between stimuli using his category" % (self, agent.id))
        logging.debug("Hearer plays the discrimination game")
        # TODO discrimination_game czy discriminate?
        try:
            category = agent.discrimination_game(context, topic)
        except NO_CATEGORY:
            self.on_NO_CATEGORY(agent=agent, context=context, topic=topic)
            logging.debug("Hearer plays the discrimination game")
            # TODO discrimination_game czy discriminate?
            category = agent.discrimination_game(context, topic)
            # raise Exception("Wow, there should be no error here.")
        except NO_NOTICEABLE_DIFFERENCE:
            self.on_NO_NOTICEABLE_DIFFERENCE()
            logging.debug("Hearer plays the discrimination game")
            # TODO discrimination_game czy discriminate?
            category = agent.discrimination_game(context, topic)
            # raise Exception("Wow, there should be no error here.")
        except NO_POSITIVE_RESPONSE_1:
            self.on_NO_POSITIVE_RESPONSE_1(agent=agent, agent_category=None, context=context)
            logging.debug("Hearer plays the discrimination game")
            # TODO discrimination_game czy discriminate?
            category = agent.discrimination_game(context, topic)
            # raise Exception("Wow, there should be no error here.")
        except NO_POSITIVE_RESPONSE_2:
            self.on_NO_POSITIVE_RESPONSE_2(agent=agent, agent_category=None, context=context)
            logging.debug("Hearer plays the discrimination game")
            # TODO discrimination_game czy discriminate?
            category = agent.discrimination_game(context, topic)
            # raise Exception("Wow, there should be no error here.")
        except NO_DISCRIMINATION_LOWER_1:
            self.on_NO_DISCRIMINATION_LOWER_1(agent=agent, agent_category=None, context=context)
            logging.debug("Hearer plays the discrimination game")
            # TODO discrimination_game czy discriminate?
            category = agent.discrimination_game(context, topic)
            # raise Exception("Wow, there should be no error here.")
        except NO_DISCRIMINATION_LOWER_2:
            self.on_NO_DISCRIMINATION_LOWER_2(agent=agent, agent_category=None, context=context)
            logging.debug("Hearer plays the discrimination game")
            # TODO discrimination_game czy discriminate?
            category = agent.discrimination_game(context, topic)
            # raise Exception("Wow, there should be no error here.")

        # learned_category = self.hearer.learn_stimulus(self.hearer_category, self.context, self.topic)
        # if learned_category != self.hearer_category:
        logging.debug("%s(%d) associates \"%s\" with category %d" % (agent,
                                                                     agent.id,
                                                                     speaker_word,
                                                                     category + 1))
        agent.learn_word_category(speaker_word, category)
