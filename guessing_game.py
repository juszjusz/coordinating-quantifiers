from __future__ import division  # force python 3 division in python 2
import logging
from agent import Agent
from language import Language
from perception import Perception
from perception import Stimulus
from random import choice


class GuessingGame:

    def __init__(self, speaker, hearer):
        self.speaker = speaker
        self.hearer = hearer
        self.in_progress = True
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

    # TODO problems, many errors can occur both at the speaker and hearer level
    def on_error(self, error, role):
        if error == Agent.Error.NO_ERROR:
            return

        self.in_progress = False
        agent = self.speaker if role == Agent.Role.SPEAKER else self.hearer
        agent_role = "Speaker" if role == Agent.Role.SPEAKER else "Hearer"
        agent_category = self.speaker_category if role == Agent.Role.SPEAKER else self.hearer_category

        if error == Perception.Error.NO_CATEGORY:
            logging.debug("no category")
            logging.debug("%s(%d)" % (agent_role, agent.id))
            agent.learn_stimulus(None, self.context, self.topic)
            return

        if error == Perception.Error.NO_NOTICEABLE_DIFFERENCE:
            logging.debug("no noticeable difference")
            return

        # TODO encapsulate
        if error == Perception.Error.NO_DISCRIMINATION_LOWER_1:
            logging.debug("no discrimination lower 1")
            logging.debug("%s(%d)" % (agent_role, agent.id))
            agent.learn_stimulus(agent_category, self.context, 0)
            return

        if error == Perception.Error.NO_DISCRIMINATION_LOWER_2:
            logging.debug("no discrimination lower 2")
            logging.debug("%s(%d)" % (agent_role, agent.id))
            agent.learn_stimulus(agent_category, self.context, 1)
            return

        if error == Perception.Error.NO_POSITIVE_RESPONSE_1:
            logging.debug("no responsive category for stimulus 1")
            logging.debug("%s(%d)" % (agent_role, agent.id))
            agent.learn_stimulus(agent_category, self.context, 0)
            return

        if error == Perception.Error.NO_POSITIVE_RESPONSE_2:
            logging.debug("no responsive category for stimulus 2")
            logging.debug("%s(%d)" % (agent_role, agent.id))
            agent.learn_stimulus(agent_category, self.context, 1)
            return

        # TODO talk to Franek ("category capable of discriminating the topic")
        if error == Language.Error.NO_WORD_FOR_CATEGORY:
            logging.debug("%s(%d) has no word for his category" % (agent_role, agent.id))
            new_word = agent.add_new_word()
            # TODO speaker_word instead new_word_index?
            logging.debug("%s(%d) introduces new word \"%s\"" % (agent_role, agent.id, new_word))
            agent.learn_word_category(new_word, agent_category)
            logging.debug("%s(%d) associates \"%s\" with his category" % (agent_role, agent.id, new_word))
            return

        # TODO added recursion, invocation of on_error, check
        if error == Language.Error.NO_SUCH_WORD:
            logging.debug("Hearer(%d) adds word \"%s\"" % (self.hearer.id, self.speaker_word))
            self.hearer.add_word(self.speaker_word)
            logging.debug("Hearer(%d) plays the discrimination game" % self.hearer.id)
            self.hearer_category, hearer_error = self.hearer.discriminate(self.context, self.topic)
            logging.debug("Hearer(%d) category %d" % (self.hearer.id,
                                                      -1 if self.hearer_category is None else self.hearer_category))
            if hearer_error != Perception.Error.NO_ERROR:
                self.on_error(hearer_error, Agent.Role.HEARER)
                logging.debug("Hearer(%d) plays the discrimination game" % self.hearer.id)
                self.hearer_category, hearer_error = self.hearer.discriminate(self.context, self.topic)
                logging.debug("Hearer(%d) category %d" % (self.hearer.id,
                                                          -1 if self.hearer_category is None else self.hearer_category))
                if hearer_error != Perception.Error.NO_ERROR:
                    raise Exception("Wow, there should be no error here.")

            logging.debug("Hearer(%d) associates \"%s\" with his category %d" % (
            self.hearer.id, self.speaker_word, self.hearer_category))
            self.hearer.learn_word_category(self.speaker_word, self.hearer_category)
            return

        # TODO no difference works for speaker and haerer?
        if error == Perception.Error.NO_DIFFERENCE_FOR_CATEGORY:
            logging.debug("%s(%d) sees no difference between stimuli using his category" % (agent_role, agent.id))
            logging.debug("Hearer plays the discrimination game")
            self.hearer_category, error = self.hearer.discriminate(self.context, self.topic)
            if error != Perception.Error.NO_ERROR:
                self.on_error(error, Agent.Role.HEARER)
                logging.debug("Hearer plays the discrimination game")
                self.hearer_category, error = self.hearer.discriminate(self.context, self.topic)
                logging.debug("Hearer category %d" % -1 if self.hearer_category is None else self.hearer_category)
                if error != Perception.Error.NO_ERROR:
                    raise Exception("Wow, there should be no error here.")

            # learned_category = self.hearer.learn_stimulus(self.hearer_category, self.context, self.topic)
            # if learned_category != self.hearer_category:
            logging.debug("%s(%d) associates \"%s\" with category %d" % (agent_role,
                                                                         agent.id,
                                                                         self.speaker_word,
                                                                         self.hearer_category))
            self.hearer.learn_word_category(self.speaker_word, self.hearer_category)
            return

    # guessing game
    def play(self):
        logging.debug("--")
        logging.debug(
            "Stimulus 1: %d/%d = %f" % (self.context[0].a, self.context[0].b, self.context[0].a / self.context[0].b))
        logging.debug(
            "Stimulus 2: %d/%d = %f" % (self.context[1].a, self.context[1].b, self.context[1].a / self.context[1].b))
        logging.debug("topic = %d" % (self.topic + 1))

        self.speaker_category, error = self.speaker.discriminate(self.context, self.topic)
        self.speaker.store_ds_result(self.speaker.Result.SUCCESS if error == Agent.Error.NO_ERROR
                                     else self.speaker.Result.FAILURE)
        logging.debug("discrimination success" if error == Agent.Error.NO_ERROR else "discrimination failure")
        self.on_error(error, Agent.Role.SPEAKER)

        if self.in_progress:
            self.speaker_word, error = self.speaker.get_word(self.speaker_category)
            msg = "nothing" if self.speaker_word is None else self.speaker_word
            logging.debug("Speaker(%d) says %s" % (self.speaker.id, msg))
            self.on_error(error, Agent.Role.SPEAKER)

        if self.in_progress:
            self.hearer_category, error = self.hearer.get_category(self.speaker_word)
            self.on_error(error, Agent.Role.HEARER)

        if self.in_progress:
            self.hearer_topic, error = self.hearer.get_topic(self.context, self.hearer_category)
            self.on_error(error, Agent.Role.HEARER)

        self.completed = self.in_progress
        success = self.topic == self.hearer_topic

        if self.completed and success:
            logging.debug("guessing game success!")
            self.speaker.store_cs_result(Agent.Result.SUCCESS)
            self.hearer.store_cs_result(Agent.Result.SUCCESS)
        else:
            logging.debug("guessing game failed!")
            self.speaker.store_cs_result(Agent.Result.FAILURE)
            self.hearer.store_cs_result(Agent.Result.FAILURE)

        # TODO if completed? - check this
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

    def get_stats(self):
        return None

# my comments:
# 1. Lets make method naming consistent, i.e. if a method is used for queries, then
# name it 'get_' + SUFFIX (or 'select_' + SUFFIX, but let it be standarized)
# 2. Follow python' naming convention (underscres instead of camelcase, i.e. 'get_word' instead of 'getWord')
# 3. Make naming agnostic with respect to implementation ('updateDictionary' instead of 'updateAssocationMatrix')
