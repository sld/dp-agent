# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from logging import getLogger
from deeppavlov.core.models.estimator import Component

logger = getLogger(__name__)


class Top1Retriever(Component):
    def __init__(self, **kwargs):
        pass

    def __call__(self, text_batch, y_proba):
        """
        Get batch of contexts, retrieve their corresponded responses,
        return batch of lists of candidates and batch of model inputs.

        context: List[List[str]]
        index: List[List[str]]
        """
        logger.info("{} {}".format(text_batch, y_proba))
        return text_batch[0], y_proba[0]
