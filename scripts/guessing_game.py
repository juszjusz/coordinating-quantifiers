# from objects.agent import SpeakerAgent, HearerAgent
from objects.agent import Agent
from objects.perception import Stimulus
from random import choice


class GuessingGame:

    def __init__(self, speaker, hearer):
        self.speaker = speaker
        self.hearer = hearer
        self.in_progress = True
        self.context = [Stimulus(), Stimulus()]
        self.speaker_topic = choice([0, 1])
        self.hearer_topic = None
        self.speaker_category = None
        self.speaker_word = None
        self.hearer_category = None
        self.hearer_word = None

    def on_error(self, error):
        if error == Agent.Error.NO_CATEGORY:
            self.speaker.learn_topic(None, self.context, self.speaker_topic)
        elif error == Agent.Error.NO_DISCRIMINATION:
            self.speaker.learn_topic(self.speaker_category, self.context, self.speaker_topic)

        # TODO talk to Franek ("category capable of discriminating the topic")
        if error == Agent.Error.NO_WORD_FOR_CATEGORY:
            new_word = self.speaker.add_new_word()
            # TODO speaker_word instead new_word_index?
            self.speaker.learn_word_category(new_word, self.hearer_category)

        if error == Agent.Error.NO_SUCH_WORD:
            self.hearer.add_word(self.speaker_word)

            self.hearer_category, error = self.hearer.discriminate(self.context, self.speaker_topic)
            if error == Agent.Error.NO_CATEGORY:
                self.hearer_category = self.hearer.learn_topic(None, self.context, self.speaker_topic)
            elif error == Agent.Error.NO_DISCRIMINATION:
                self.hearer_category = self.hearer.learn_topic(self.speaker_category, self.context, self.speaker_topic)

            self.hearer.learn_word_category(self.speaker_word, self.hearer_category)

        if error == Agent.Error.NO_DIFFERENCE:
            # TODO
            raise Exception("no difference between stimuli")

    # guessing game
    def play(self):

        self.speaker_category, error = self.speaker.discriminate(self.context, self.speaker_topic)
        self.on_error(error)

        if self.in_progress:
            self.speaker_word, error = self.speaker.get_word(self.speaker_category)
            self.on_error(error)

        if self.in_progress:
            self.hearer_category, error = self.hearer.get_category(self.speaker_word)
            self.on_error(error)

        if self.in_progress:
            self.hearer_topic, error = self.hearer.get_topic(self.context, self.hearer_category)
            self.on_error(error)

        success = self.speaker_topic == self.hearer_topic

        if self.speaker_category is not None and self.hearer_category is not None and self.speaker_word is not None:
            self.speaker.update(success=success, role=Agent.Role.SPEAKER,
                                word=self.speaker_word, category=self.speaker_category)
            self.hearer.update(success=success, role=Agent.Role.HEARER,
                               word=self.speaker_word, category=self.hearer_category)

        # TODO stage 7
        #print('goto to stage 7')

# my comments:
# 1. Lets make method naming consistent, i.e. if a method is used for queries, then
# name it 'get_' + SUFFIX (or 'select_' + SUFFIX, but let it be standarized)
# 2. Follow python' naming convention (underscres instead of camelcase, i.e. 'get_word' instead of 'getWord')
# 3. Make naming agnostic with respect to implementation ('updateDictionary' instead of 'updateAssocationMatrix')