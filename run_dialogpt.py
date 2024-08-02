import json
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, BlenderbotTokenizer, BlenderbotForConditionalGeneration, BitsAndBytesConfig
from transformers import TextGenerationPipeline, AutoModelWithLMHead, pipeline
import torch
import os
import openai
import abc
import time
import urllib
import requests
from requests.structures import CaseInsensitiveDict
import ast
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.nlp.v20190408 import nlp_client, models
from cleverwrap import CleverWrap

class Bot(abc.ABC):
    @abc.abstractclassmethod
    def respond(self, utterance):
        pass

    def interact(self):
        while True:
            UTTERANCE = input("You: ")
            print(f"Bot: {self.respond(UTTERANCE)}")     

    def script(self, questions):
        responses = []
        for question in questions:
            responses.append(self.respond(question))
        return responses
            

class DialoGPT(Bot):
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

    def respond(self, utterance):
        new_user_input_ids = self.tokenizer.encode(utterance + self.tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = new_user_input_ids
        chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
