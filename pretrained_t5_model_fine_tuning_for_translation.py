# -*- coding: utf-8 -*-
"""Pretrained T5 model  fine-tuning """

!sudo pip install portalocker
import torch
import torch.nn as nn
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k, multi30k
from typing import Iterable, List

"""Download stuff.."""

!sudo pip install -U torchdata
!sudo pip install -U spacy
!sudo python -m spacy download en_core_web_sm
!sudo python -m spacy download de_core_news_sm

"""Dataset"""

language_pair = ("en", "de")
train, valid, test = Multi30k(language_pair=language_pair)
BATCH_SIZE = 2
task = "translate English to German"

def apply_prefix(task,x):
  return f"{task}: " + x[0], x[1]

from torchtext.vocab.vectors import partial
# Добавить к строке задание пример: "перевод с анг на нем"
step_1 = valid.map(partial(apply_prefix,task))

#Сколько одноваременно будет обрабатываться трансфрмером
step_2 = step_1.batch(BATCH_SIZE)

# Добавение тегов к каждому типу. Также делает каждый batch итерируемым
step_3 = step_2.rows2columnar(["english", "german"])

#Загружает в тр в правильном виде, можно разбить на батчи но мы это сделали уже в 2 шаге
from torch.utils.data import DataLoader
step_4  = DataLoader(step_3, batch_size=None)

for toens in step_4:
  print(toens)

#Загружаем T5 модель которая уже была натренерована
#https://arxiv.org/pdf/1910.10683.pdf
from torchtext.models import T5_BASE_GENERATION
t5_base = T5_BASE_GENERATION
transform = t5_base.transform()
model = t5_base.get_model()
model.eval()

batch = next(iter(step_4)) #Получает 1 батчи

input_text = batch["english"]
target = batch["german"]

input_text

model_input = transform(input_text)

model_input

from torchtext.prototype.generate import GenerationUtils
sequence_generator = GenerationUtils(model)

model_output = sequence_generator.generate(model_input, eos_idx=1, num_beams=1)

output_text = transform.decode(model_output.tolist())

print(output_text)
