from datetime import datetime
import uuid

from core.agent import Agent
from core.state_manager import StateManager
from core.skill_manager import SkillManager
from core.rest_caller import RestCaller
from core.postprocessor import DefaultPostprocessor
from core.response_selector import ConfidenceResponseSelector
from core.transform_config import MAX_WORKERS, ANNOTATORS, SKILL_SELECTORS
from core.skill_selector import ChitchatQASelector
from core.state_schema import Human
# from deeppavlov import configs, build_model

# ner = build_model(configs.ner.ner_rus, download=True)
# faq = build_model(configs.faq.tfidf_autofaq, download=True)
# sentiment = build_model(configs.classifiers.rusentiment_elmo_twitter_rnn, download=True)
# utterances = ['Привет!', 'Когда началась Вторая Мировая?',
#               'Привет, я бот!', '1939', 'Как дела?', 'Спасибо, бот!',
#               'Хорошо, а у тебя как?', 'И у меня нормально. Когда родился Петр Первый?',
#               'в 1672 году', 'спасибо', ]
# print("DeepPavlov configs output:")
# print(ner(utterances))
# print(faq(utterances))
# print(sentiment(utterances))

state_manager = StateManager()

anno_names, anno_urls = zip(*[(annotator['name'], annotator['url']) for annotator in ANNOTATORS])
preprocessor = Service(
    rest_caller=RestCaller(max_workers=MAX_WORKERS, names=anno_names, urls=anno_urls))
postprocessor = DefaultPostprocessor()

skill_caller = RestCaller(max_workers=MAX_WORKERS)
response_selector = ConfidenceResponseSelector()
ss_names, ss_urls = zip(*[(selector['name'], selector['url']) for selector in SKILL_SELECTORS])
skill_selector = ChitchatQASelector(RestCaller(max_workers=MAX_WORKERS, names=ss_names, urls=ss_urls))
skill_manager = SkillManager(skill_selector=skill_selector, response_selector=response_selector,
                             skill_caller=skill_caller)

agent = Agent(state_manager, preprocessor, postprocessor, skill_manager)

# TEST predict_annotations()
# annotations = agent.predict_annotations(utterances, should_reset=[False]*len(utterances))
# print("Agent output:")
# print(annotations)

# TEST __call__()
exist_humans = Human.objects
u_tg_ids = [exist_humans[0].user_telegram_id, exist_humans[1].user_telegram_id, str(uuid.uuid4())]
utts = ['Что еще скажешь интересного?', 'Бот, ты тупой', '/start']
u_d_types = ['iphone', 'android', 'iphone']
date_times = [datetime.utcnow(), datetime.utcnow(), datetime.utcnow()]
locations = ['moscow', 'novosibirsk', 'novokuznetsk']
ch_types = ['telegram', 'telegram', 'telegram']

responses = agent(utterances=utts, user_telegram_ids=u_tg_ids, user_device_types=u_d_types,
                  date_times=date_times, locations=locations, channel_types=ch_types)
print(responses)
