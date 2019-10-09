from __future__ import division  # force python 3 division in python 2
import logging
from agent import Agent, Hearer, Speaker
from guessing_game_exceptions import *
from random import choice


class GuessingGame:

    def __init__(self, is_stage7_on, context):
        self.completed = False
        self.is_stage7_on = is_stage7_on
        self.context = context
        self.topic = choice([0, 1])
        self.exception_handler = ExceptionHandler()

    # guessing game
    def play(self, speaker, hearer):
        logging.debug("--")
        logging.debug(
            "Stimulus 1: %s" % self.context[0])
        logging.debug(
            "Stimulus 2: %s" % self.context[1])
        logging.debug("topic = %d" % (self.topic + 1))

        hearer_topic = None
        speaker_category = None
        speaker_word = None
        hearer_category = None
        hearer_category2 = None

        try:
            speaker_category = speaker.discrimination_game(self.context, self.topic)
            logging.debug("Speaker(%d)'s discriminative category: %d" % (speaker.id, speaker.get_categories()[speaker_category].id))
            speaker_word = speaker.get_most_connected_word(speaker_category)
            logging.debug("Speaker(%d) says: %s" % (speaker.id, speaker_word))
            hearer_category = hearer.get_most_connected_category(speaker_word)
            logging.debug("Hearer(%d)'s category: %d" % (hearer.id, hearer.get_categories()[hearer_category].id))
            hearer_topic = hearer.get_topic(context=self.context, category=hearer_category)
            logging.debug("Topic according to hearer(%d): %d" % (hearer.id, hearer_topic+1))
            self.completed = True
        except NO_CATEGORY:
            self.exception_handler.on_NO_CATEGORY(agent=speaker, context=self.context, topic=self.topic)
        except NO_NOTICEABLE_DIFFERENCE:
            self.exception_handler.on_NO_NOTICEABLE_DIFFERENCE()
        except NO_POSITIVE_RESPONSE_1:
            self.exception_handler.on_NO_POSITIVE_RESPONSE_1(agent=speaker,
                                                             context=self.context,
                                                             category_index=speaker_category)
        except NO_POSITIVE_RESPONSE_2:
            self.exception_handler.on_NO_POSITIVE_RESPONSE_2(agent=speaker,
                                                             context=self.context,
                                                             category_index=speaker_category)
        except NO_DISCRIMINATION_LOWER_1 as e:
            self.exception_handler.on_NO_DISCRIMINATION_LOWER_1(agent=speaker,
                                                                context=self.context,
                                                                category_index=e.i)
        except NO_DISCRIMINATION_LOWER_2 as e:
            self.exception_handler.on_NO_DISCRIMINATION_LOWER_2(agent=speaker,
                                                                context=self.context,
                                                                category_index=e.i)
        except NO_WORD_FOR_CATEGORY:
            self.exception_handler.on_NO_WORD_FOR_CATEGORY(speaker=speaker, agent_category=speaker_category)
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
            hearer_category = self.exception_handler.on_NO_SUCH_WORD(hearer=hearer, context=self.context,
                                                                     topic=self.topic,
                                                                     speaker_word=speaker_word)
        except ERROR:
            self.exception_handler.on_LANGUAGE_ERROR()

        # logging.debug("discrimination success" if error == Agent.Error.NO_ERROR else "discrimination failure")

        success1 = self.topic == hearer_topic

        if success1:
            logging.debug("guessing game 1 success!")
        else:
            logging.debug("guessing game 1 failed!")

        speaker.store_cs_result(Agent.Result.SUCCESS if success1 else Agent.Result.FAILURE)
        hearer.store_cs_result(Agent.Result.SUCCESS if success1 else Agent.Result.FAILURE)

        if self.completed and success1:
            speaker.update_on_success(speaker_word, speaker_category)
            hearer.update_on_success(speaker_word, hearer_category)
        elif self.completed:
            hearer.update_on_failure(speaker_word, hearer_category)
            speaker.update_on_failure(speaker_word, speaker_category)

        success2 = False
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
            except NO_POSITIVE_RESPONSE_1:
                self.exception_handler.on_NO_POSITIVE_RESPONSE_1(agent=hearer,
                                                                 context=self.context,
                                                                 category_index=hearer_category2)
            except NO_POSITIVE_RESPONSE_2:
                self.exception_handler.on_NO_POSITIVE_RESPONSE_2(agent=hearer,
                                                                 context=self.context,
                                                                 category_index=hearer_category2)
            except NO_DISCRIMINATION_LOWER_1 as e:
                self.exception_handler.on_NO_DISCRIMINATION_LOWER_1(agent=hearer,
                                                                    context=self.context,
                                                                    category_index=e.i)
            except NO_DISCRIMINATION_LOWER_2 as e:
                self.exception_handler.on_NO_DISCRIMINATION_LOWER_2(agent=hearer,
                                                                    context=self.context,
                                                                    category_index=e.i)
            except ERROR:
                self.exception_handler.on_LANGUAGE_ERROR()


            success2 = word == speaker_word

            logging.debug("Hearer(%d) says %s" % (hearer.id, word))
            #speaker.store_cs2_result(Agent.Result.SUCCESS if success2 else Agent.Result.FAILURE)
            #hearer.store_cs2_result(Agent.Result.SUCCESS if success2 else Agent.Result.FAILURE)

            if success2:
                logging.debug("guessing game 2 success!")
                speaker.update_on_success_stage7(speaker_word, speaker_category)
                hearer.update_on_success_stage7(word, word_categories)
            else:
                logging.debug("guessing game 2 failed!")

            speaker.language.forget_words()
            hearer.language.forget_words()

class ExceptionHandler:
    # to be move to Speaker Hearer subclass
    def on_NO_CATEGORY(self, agent, context, topic):
        logging.debug("no category")
        logging.debug("%s(%d)" % (agent, agent.id))
        agent.learn_stimulus(context, topic)

    # TODO wyrzucic?
    # to be move to Speaker Hearer subclass
    def on_NO_NOTICEABLE_DIFFERENCE(self):
        logging.debug("no noticeable difference")
        # to be move to Speaker Hearer subclass

    def on_NO_DISCRIMINATION_LOWER_1(self, agent, context, category_index):
        logging.debug("no discrimination lower 1")
        logging.debug("%s(%d)" % (agent, agent.id))
        agent.learn_stimulus(context, 0, category_index)

    # to be move to Speaker Hearer subclass
    def on_NO_DISCRIMINATION_LOWER_2(self, agent, context, category_index):
        logging.debug("no discrimination lower 2")
        logging.debug("%s(%d)" % (agent, agent.id))
        agent.learn_stimulus(context, 1, category_index)

    # to be move to Speaker Hearer subclass
    def on_NO_POSITIVE_RESPONSE_1(self, agent, context, category_index):
        logging.debug("no responsive category for stimulus 1")
        logging.debug("%s(%d)" % (agent, agent.id))
        agent.learn_stimulus(context, 0, category_index)

    # to be move to Speaker Hearer subclass
    def on_NO_POSITIVE_RESPONSE_2(self, agent, context, category_index):
        logging.debug("no responsive category for stimulus 2")
        logging.debug("%s(%d)" % (agent, agent.id))
        agent.learn_stimulus(context, 1, category_index)

    # to be move to Speaker and Hearer subclass
    def on_LANGUAGE_ERROR(self):
        logging.debug("Unknown error")
        pass

    # to be move to Speaker subclass
    def on_NO_WORD_FOR_CATEGORY(self, speaker, agent_category):
        logging.debug("%s(%d) has no word for his category" % (speaker, speaker.id))
        new_word = speaker.add_new_word()
        # TODO speaker_word instead new_word_index?
        logging.debug("%s(%d) introduces new word \"%s\"" % (speaker, speaker.id, new_word))
        speaker.learn_word_category(new_word, agent_category)
        logging.debug("%s(%d) associates \"%s\" with his category" % (speaker, speaker.id, new_word))

    # to be move to Hearer subclass
    def on_NO_SUCH_WORD(self, hearer, context, topic, speaker_word):
        logging.debug("Hearer(%d) adds word \"%s\"" % (hearer.id, speaker_word))
        hearer.add_word(speaker_word)
        logging.debug("Hearer(%d) plays the discrimination game" % hearer.id)
        # TODO discrimination_game czy discriminate?
        category = None
        try:
            category = hearer.discrimination_game(context, topic)
            logging.debug("Hearer(%d) category %d" % (hearer.id,
                                                      -1 if category is None else hearer.get_categories()[category].id))
            logging.debug("Hearer(%d) associates \"%s\" with his category %d" % (
                hearer.id, speaker_word, hearer.get_categories()[category].id))
            hearer.learn_word_category(speaker_word, category)
            return category
        except NO_CATEGORY:
            self.on_NO_CATEGORY(agent=hearer, context=context, topic=topic)
            # TODO discuss
        except NO_NOTICEABLE_DIFFERENCE:
            self.on_NO_NOTICEABLE_DIFFERENCE()
            # TODO do wywalenia?
            # raise Exception("Wow, there should be no error here.")
        except NO_POSITIVE_RESPONSE_1:
            self.on_NO_POSITIVE_RESPONSE_1(agent=hearer, context=context, category_index=category)
            # TODO discuss
            # raise Exception("Wow, there should be no error here.")
        except NO_POSITIVE_RESPONSE_2:
            self.on_NO_POSITIVE_RESPONSE_2(agent=hearer, context=context, category_index=category)
            # TODO discuss
            # raise Exception("Wow, there should be no error here.")
        except NO_DISCRIMINATION_LOWER_1:
            self.on_NO_DISCRIMINATION_LOWER_1(agent=hearer, context=context, category_index=category)
            # TODO discuss
            # raise Exception("Wow, there should be no error here.")
        except NO_DISCRIMINATION_LOWER_2:
            self.on_NO_DISCRIMINATION_LOWER_2(agent=hearer, context=context, category_index=category)
            # TODO discuss
            # raise Exception("Wow, there should be no error here.")

        logging.debug("Hearer(%d) category %d" % (hearer.id, -1))
        logging.debug("Hearer(%d) plays the discrimination game" % hearer.id)

        try:
            category = hearer.discrimination_game(context, topic)
            logging.debug("Hearer(%d) category %d" % (hearer.id,
                                                      -1 if category is None else hearer.get_categories()[category].id))
            logging.debug("Hearer(%d) associates \"%s\" with his category %d" % (
                hearer.id, speaker_word, hearer.get_categories()[category].id))
            hearer.learn_word_category(speaker_word, category)
            return category
        except PerceptionError:
            logging.debug("Hearer(%d) - discrimination failure" % hearer.id)
            logging.debug("Hearer(%d) does not associate \"%s\" with any category" % (hearer.id, speaker_word))
            return None

    # HEArER
    def on_NO_DIFFERENCE_FOR_CATEGORY(self, hearer, context, topic, speaker_word):
        logging.debug("Hearer(%d) sees no difference between stimuli using his category" % (hearer.id))
        logging.debug("Hearer plays the discrimination game")
        # TODO discrimination_game czy discriminate?
        category = None
        try:
            category = hearer.discrimination_game(context, topic)
            logging.debug("Hearer(%d) associates \"%s\" with category %d" % (hearer.id, speaker_word, hearer.get_categories()[category].id))
            hearer.learn_word_category(speaker_word, category)
            return category
        except NO_CATEGORY:
            self.on_NO_CATEGORY(agent=hearer, context=context, topic=topic)
            # raise Exception("Wow, there should be no error here.")
        except NO_NOTICEABLE_DIFFERENCE:
            self.on_NO_NOTICEABLE_DIFFERENCE()
            # raise Exception("Wow, there should be no error here.")
        except NO_POSITIVE_RESPONSE_1:
            self.on_NO_POSITIVE_RESPONSE_1(agent=hearer, context=context, category_index=category)
            # raise Exception("Wow, there should be no error here.")
        except NO_POSITIVE_RESPONSE_2:
            self.on_NO_POSITIVE_RESPONSE_2(agent=hearer, context=context, category_index=category)
            # raise Exception("Wow, there should be no error here.")
        except NO_DISCRIMINATION_LOWER_1:
            self.on_NO_DISCRIMINATION_LOWER_1(agent=hearer, context=context, category_index=category)
            # raise Exception("Wow, there should be no error here.")
        except NO_DISCRIMINATION_LOWER_2:
            self.on_NO_DISCRIMINATION_LOWER_2(agent=hearer, context=context, category_index=category)
            # raise Exception("Wow, there should be no error here.")

        try:
            logging.debug("Hearer plays the discrimination game")
            # TODO discrimination_game czy discriminate?
            category = hearer.discrimination_game(context, topic)
            logging.debug("Hearer(%d) associates \"%s\" with category %d" % (hearer.id, speaker_word, hearer.get_categories()[category].id))
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
            logging.debug("Hearer(%d) associates \"%s\" with category %d" % (hearer.id, speaker_word, hearer.get_categories()[category].id))
            hearer.learn_word_category(speaker_word, category)
            return category
        except NO_CATEGORY:
            self.on_NO_CATEGORY(agent=hearer, context=context, topic=topic)
            # raise Exception("Wow, there should be no error here.")
        except NO_NOTICEABLE_DIFFERENCE:
            self.on_NO_NOTICEABLE_DIFFERENCE()
            # raise Exception("Wow, there should be no error here.")
        except NO_POSITIVE_RESPONSE_1:
            self.on_NO_POSITIVE_RESPONSE_1(agent=hearer, context=context, category_index=category)
            # raise Exception("Wow, there should be no error here.")
        except NO_POSITIVE_RESPONSE_2:
            self.on_NO_POSITIVE_RESPONSE_2(agent=hearer, context=context, category_index=category)
            # raise Exception("Wow, there should be no error here.")
        except NO_DISCRIMINATION_LOWER_1:
            self.on_NO_DISCRIMINATION_LOWER_1(agent=hearer, context=context, category_index=category)
            # raise Exception("Wow, there should be no error here.")
        except NO_DISCRIMINATION_LOWER_2:
            self.on_NO_DISCRIMINATION_LOWER_2(agent=hearer, context=context, category_index=category)
            # raise Exception("Wow, there should be no error here.")

        logging.debug("Discrimination game failed")
        try:
            logging.debug("Hearer plays the discrimination game")
            # TODO discrimination_game czy discriminate?
            category = hearer.discrimination_game(context, topic)
            logging.debug("Hearer(%d) associates \"%s\" with category %d" % (hearer.id, speaker_word, hearer.get_categories()[category].id))
            hearer.learn_word_category(speaker_word, category)
            return category
        except PerceptionError:
            logging.debug("Hearer is unable to discriminate the topic")
            return None
