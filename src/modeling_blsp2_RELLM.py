import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

import logging
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import WhisperConfig


from .plora import LoraConfig, LoraModel
from .modeling_adapter_w2v2cat import Subsampler, CFormer
from .configuration_blsp2 import Blsp2Config
from .configuration_qwen import QWenConfig
from .modeling_utils import length_to_attention_mask, check_shape
from .modeling_whisper_encoder import WhisperEncoder
from .modeling_qwen import QWenLMHeadModel
from .modeling_w2v2adapter import LinearProjector
import torch.nn.functional as F

### module from w2v2 ###
import numpy as np
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import librosa
from .modeling_w2v2 import EmotionModel, process_func_revised_resample

### module: get w2v2_embed
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

text_llm_related_losses = {"response_kl", "input_kl"}
speech_llm_related_losses = {"response_kl", "input_kl", "response_ce", "input_er", "input_er_w2v2", "input_er_avd"}
lm_related_losses = text_llm_related_losses | speech_llm_related_losses


class Blsp2Model(PreTrainedModel):
    config_class = Blsp2Config
    base_model_prefix = "blsp2"

    def __init__(self, config: Blsp2Config):
        super().__init__(config)
        self.whisper_config = WhisperConfig(**config.whisper_config)
        self.qwen_config = QWenConfig(**config.qwen_config)

        self.whisper_model = WhisperEncoder(self.whisper_config)
        self.qwen_model = QWenLMHeadModel(self.qwen_config)

        ### module from w2v2 ###

        w2v2_name = '/home/jinghan/Desktop/w2v2-how-to/model/wav2vec2-large-robust-12-ft-emotion-msp-dim/'
        self.processor = Wav2Vec2Processor.from_pretrained(w2v2_name)
        self.w2v2 = EmotionModel.from_pretrained(w2v2_name)
        self.w2v2dim = 1024
        self.w2v2_adapter = LinearProjector(self.w2v2dim, config.adapter_inner_dim, self.qwen_config.hidden_size)
        self.w2v2_hidden2emotion = nn.Linear(self.qwen_config.hidden_size, self.config.num_emotions, bias=False)

        self.num_w2v2 = 3
        self.hidden2w2v2 = nn.Linear(self.qwen_config.hidden_size, self.num_w2v2, bias=False)
        ########################

        #### module for speechcatw2v2 ####
        #print(f"in init:{self.whisper_config.d_model}")
        self.w2v2cat_dim = self.w2v2dim + self.whisper_config.d_model
        self.w2v2cat_adapter = Subsampler(self.w2v2cat_dim, config.adapter_inner_dim, self.qwen_config.hidden_size,
                                      config.adapter_hidden_layers, self.whisper_config, config.conv_kernel_sizes)
        self.w2v2cat_adapter.init_weights_forw2v2cat()
        ########################

        ### module for w2v2_emo direct input ###
        self.emotion2idx = {
            "neutral": 0,
            "happy": 1,
            "angry": 2,
            "sad": 3,
            "surprise": 4,
            "":5
        }
        self.w2v2dict = {}
        w2v2_embpath = "/home/jinghan/Desktop/blsp-emo/examples/test/extractw2v2_from_output_my_sess5_iemocap_w2v2cat_pv2_epo3_embedsave.jsonl"
        self.load_w2v2embed(w2v2_embpath)
        ###

        if config.lora_config:
            self.lora_config = LoraConfig(**config.lora_config)
            self.qwen_model = LoraModel(self.qwen_model, self.lora_config, "default")

        if config.adapter_type == "subsampler":
            self.adapter = Subsampler(self.whisper_config.d_model, config.adapter_inner_dim, self.qwen_config.hidden_size,
                                      config.adapter_hidden_layers, self.whisper_config, config.conv_kernel_sizes)

        elif config.adapter_type == "cformer":
            self.adapter = CFormer(self.whisper_config, self.qwen_config.hidden_size,
                                   self.qwen_config.vocab_size,
                                   num_pre_cif_layers=config.num_pre_cif_layers,
                                   num_post_cif_layers=config.num_post_cif_layers)
        else:
            raise ValueError(f"unsupported adapter type: {config.adapter_type}")

        self.hidden2emotion = nn.Linear(self.qwen_config.hidden_size, self.config.num_emotions, bias=False)

        self.loss_names = [] # must be a list of loss names:  seq_kd, token_kd, or others before training

    def set_loss_names(self, names):
        print(f"in modeling_blsp 94:loss name:{names}")
        self.loss_names = names

    def forward(
        self,
        start_ids: torch.LongTensor,
        start_mask: torch.Tensor,
        start_labels: torch.LongTensor,
        instruction_ids: torch.LongTensor,
        instruction_mask: torch.Tensor,
        instruction_labels: torch.LongTensor,
        audio_instruction_ids: torch.LongTensor,
        audio_instruction_mask: torch.Tensor,
        audio_instruction_labels: torch.LongTensor,
        w2v2_instruction_ids: torch.LongTensor,
        w2v2_instruction_mask: torch.Tensor,
        w2v2_instruction_labels: torch.LongTensor,
        input_ids: torch.LongTensor,
        input_mask: torch.Tensor,
        input_labels: torch.LongTensor,
        speech_values: torch.FloatTensor,
        speech_mask: torch.LongTensor,
        raw_speech: torch.FloatTensor,
        sampling_rate:int,
        suffix_ids: torch.LongTensor,
        suffix_mask: torch.Tensor,
        suffix_labels: torch.LongTensor,
        emotion_labels: torch.LongTensor = None,
        arousal_ans:torch.FloatTensor = None,
        valence_ans:torch.FloatTensor = None,
        dominance_ans:torch.FloatTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        assert len(self.loss_names) > 0, "self.loss_names cannot be empty"

        if not any ("response" in loss_name for loss_name in self.loss_names):
            batch_size = start_ids.size(0)
            instruction_ids = torch.zeros(batch_size, 0, dtype=start_ids.dtype, device=start_ids.device)
            instruction_mask = torch.zeros(batch_size, 0, dtype=start_mask.dtype, device=start_mask.device)
            instruction_labels = torch.zeros(batch_size, 0, dtype=start_labels.dtype, device=start_labels.device)
            audio_instruction_ids = torch.zeros(batch_size, 0, dtype=start_ids.dtype, device=start_ids.device)
            audio_instruction_mask = torch.zeros(batch_size, 0, dtype=start_mask.dtype, device=start_mask.device)
            audio_instruction_labels = torch.zeros(batch_size, 0, dtype=start_labels.dtype, device=start_labels.device)
            suffix_ids = torch.zeros(batch_size, 0, dtype=start_ids.dtype, device=start_ids.device)
            suffix_mask = torch.zeros(batch_size, 0, dtype=start_mask.dtype, device=start_mask.device)
            suffix_labels = torch.zeros(batch_size, 0, dtype=start_labels.dtype, device=start_labels.device)


        start_embeds = self.qwen_model.get_input_embeddings()(start_ids)
        instruction_embeds = self.qwen_model.get_input_embeddings()(instruction_ids)
        audio_instruction_embeds = self.qwen_model.get_input_embeddings()(audio_instruction_ids)
        input_embeds = self.qwen_model.get_input_embeddings()(input_ids)
        suffix_embeds = self.qwen_model.get_input_embeddings()(suffix_ids)

        w2v2_instruction_embeds = self.qwen_model.get_input_embeddings()(w2v2_instruction_ids)

        #w2v2_assign = self.get_w2v2emb_pretrained(emotion_labels)
        #speech_input_embeds, speech_input_mask, speech_input_logits, speech_cif_alphas, speech_pred_num_tokens = \
        #    self.get_speech_features_addw2v2(speech_values, speech_mask, w2v2_assign, input_mask.sum(-1))
        arousal, dominance, valence, emo_embed = self.get_pure_emo_features(raw_speech, sampling_rate)
        speech_input_embeds, speech_input_mask, speech_input_logits, speech_cif_alphas, speech_pred_num_tokens = \
            self.get_speech_features_addw2v2(speech_values, speech_mask, emo_embed, input_mask.sum(-1))

        #speech_input_embeds, speech_input_mask, speech_input_logits, speech_cif_alphas, speech_pred_num_tokens = \
        #    self.get_speech_features(speech_values, speech_mask, input_mask.sum(-1))
        speech_input_labels = speech_input_mask.new_ones(speech_input_embeds.size(0), speech_input_embeds.size(1),
                                                         dtype=torch.int64).fill_(-100)


        speech_embeds = torch.cat([start_embeds, audio_instruction_embeds, speech_input_embeds, suffix_embeds], dim=1)
        speech_mask = torch.cat([start_mask, audio_instruction_mask, speech_input_mask, suffix_mask], dim=1)
        speech_labels = torch.cat([start_labels, audio_instruction_labels, speech_input_labels, suffix_labels], dim=1)
        #print(f"in Blsp2Model forward/line_153:\nspeech value:{len(speech_values)} x {len(speech_values[0])} x {len(speech_values[0][0])}\n{speech_values}\nspeech input embed:{len(speech_input_embeds)} x {len(speech_input_embeds[0])} x {len(speech_input_embeds[0][0])}\n{speech_input_embeds}\nspeech_embed = size:{len(speech_embeds)}x{len(speech_embeds[0])}x{len(speech_embeds[0][0])}\n{speech_embeds}\nspeech_mask = size:{len(speech_mask)}x{len(speech_mask[0])}\n{speech_mask}\nspeech_labels = size{len(speech_labels)}x{len(speech_labels[0])}\n{speech_labels}")
        #print(f"in Blsp2Model forward:\n start_ids\n{start_ids}\n w2v2_instruction_ids\n{w2v2_instruction_ids} ")
        #print(f"in Blsp2Model forward:\n start_embeds\n{start_embeds}\n w2v2_instruction_embeds\n{w2v2_instruction_embeds} ")

        '''
        arousal, dominance, valence, emo_embed, emo_ad = \
            self.get_emo_features(raw_speech,sampling_rate, input_mask.sum(-1))
        emo_mask = torch.tensor((), dtype=torch.int64)
        emo_mask = emo_mask.new_ones(emo_ad.size(0), emo_ad.size(1),dtype=torch.int64)
        emo_labels = emo_mask.new_ones(emo_ad.size(0), emo_ad.size(1),dtype=torch.int64).fill_(-100)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        emo_mask = emo_mask.to(device)
        emo_labels = emo_labels.to(device)
        '''

        #print(f"device:\nstart_emb:{start_embeds.device}\nspeech_input_embeds\n{speech_input_embeds.device}\nemo_ad:{emo_ad.device}")
        #print(f"emo_ad:\n{len(emo_ad)}x{len(emo_ad[0])}x{len(emo_ad[0][0])}\n{emo_ad}")
        #print(f"in Blsp2Model forward: device:\nstart_mask:{start_mask.device}\nemo_mask:{emo_mask.device}\n suffix_mask:{ suffix_mask.device}")
        #print(f"in Blsp2Model forward: device:\nstart_labels:{start_labels.device}\nemo_labels:{emo_labels.device}")
        #speech_embeds = torch.cat([start_embeds, w2v2_instruction_embeds, emo_ad , audio_instruction_embeds, speech_input_embeds, suffix_embeds], dim=1)
        #speech_mask = torch.cat([start_mask, w2v2_instruction_mask, emo_mask, audio_instruction_mask, speech_input_mask, suffix_mask], dim=1)
        #speech_labels = torch.cat([start_labels, w2v2_instruction_labels, emo_labels, audio_instruction_labels, speech_input_labels, suffix_labels], dim=1)

        #print(f"in Blsp2Model forward:after adding \nspeech value:{len(speech_values)} x {len(speech_values[0])} x {len(speech_values[0][0])}\n{speech_values}\nspeech_embed = size:{len(speech_embeds)}x{len(speech_embeds[0])}x{len(speech_embeds[0][0])}\n{speech_embeds}\nspeech_mask = size:{len(speech_mask)}x{len(speech_mask[0])}\n{speech_mask}\nspeech_labels = size{len(speech_labels)}x{len(speech_labels[0])}\n{speech_labels}")
        #print(f"in Blsp2Model forward:\nraw speech:size:{len(raw_speech)} x {len(raw_speech[0])}\n{raw_speech}\n,arousal, dominance, valence = {arousal} / {dominance} / {valence} \nemo_embed:\n{emo_embed}\nemo_ad:{len(emo_ad)}x{len(emo_ad[0])}x{len(emo_ad[0][0])}\n{emo_ad}\nemo_mask:{len(emo_mask[0])}\n{emo_mask}\nemo_label:{len(emo_labels[0])}\n{emo_labels}")

        if any(loss_name in text_llm_related_losses for loss_name in self.loss_names):
            text_embeds = torch.cat([start_embeds, instruction_embeds, input_embeds, suffix_embeds], dim=1)
            text_mask = torch.cat([start_mask, instruction_mask, input_mask, suffix_mask], dim=1)
            text_labels = torch.cat([start_labels, instruction_labels, input_labels, suffix_labels], dim=1)
            input_kd_labels = torch.cat([torch.zeros_like(start_labels),
                                         torch.zeros_like(instruction_labels),
                                         input_mask,
                                         torch.zeros_like(suffix_labels)], dim=1)
            speech_kd_labels = torch.cat([torch.zeros_like(start_labels),
                                          torch.zeros_like(audio_instruction_labels),
                                          input_mask,
                                          torch.zeros_like(suffix_labels)], dim=1)
            text_response_kd_labels = torch.cat([torch.zeros_like(start_labels),
                                                 torch.zeros_like(instruction_labels),
                                                 torch.zeros_like(input_labels),
                                                 (suffix_labels != -100).long()], dim=1)
            speech_response_kd_labels = torch.cat([torch.zeros_like(start_labels),
                                                   torch.zeros_like(audio_instruction_labels),
                                                   torch.zeros_like(speech_input_labels),
                                                   (suffix_labels != -100).long()], dim=1)
            lora_audio_mask = torch.zeros_like(text_labels)
            self.update_lora_mask(lora_audio_mask, False)
            with torch.no_grad():
                text_output = self.qwen_model(inputs_embeds=text_embeds, attention_mask=text_mask,
                                              position_ids=text_mask.cumsum(dim=-1) - 1, output_hidden_states=True,
                                              return_dict=True)
                text_logits = text_output.logits

        '''
        if any(loss_name in speech_llm_related_losses for loss_name in self.loss_names):
            #print(f"in if any(loss_name in speech_llm_related_losses for loss_name in self.loss_names):")
            #print(f"lora")
            lora_audio_mask = torch.cat([torch.zeros_like(start_mask),
                                         torch.zeros_like(w2v2_instruction_mask),
                                         torch.ones_like(emo_mask),
                                         torch.zeros_like(audio_instruction_mask),
                                         torch.ones_like(speech_input_mask),
                                         torch.zeros_like(suffix_mask)], dim=1)
            self.update_lora_mask(lora_audio_mask, False)
            speech_embeds_bf16 = speech_embeds.to(torch.bfloat16)
            speech_mask_bf16 = speech_mask.to(torch.bfloat16)
            speech_output_bf16 = self.qwen_model(inputs_embeds=speech_embeds_bf16, attention_mask=speech_mask_bf16,
                                            position_ids=speech_mask_bf16.cumsum(dim=-1) - 1, output_hidden_states=True,
                                            return_dict=True)
            speech_logits_bf16 = speech_output_bf16.logits
            speech_logits = speech_logits_bf16.to(torch.float32)
            #print(f"in if any(loss_name in speech_llm_related_losses for loss_name in self.loss_names):")
            #print(f"speech_output_bf16:\n{speech_output_bf16}\nspeech_logits:\n{speech_logits}")
        '''
        '''
        if any(loss_name in speech_llm_related_losses for loss_name in self.loss_names):
            #print(f"in if any(loss_name in speech_llm_related_losses for loss_name in self.loss_names):")
            #print(f"lora")
            lora_audio_mask = torch.cat([torch.zeros_like(start_mask),
                                         torch.zeros_like(audio_instruction_mask),
                                         torch.ones_like(speech_input_mask),
                                         torch.zeros_like(suffix_mask)], dim=1)
            self.update_lora_mask(lora_audio_mask, False)
            speech_output = self.qwen_model(inputs_embeds=speech_embeds, attention_mask=speech_mask,
                                            position_ids=speech_mask.cumsum(dim=-1) - 1, output_hidden_states=True,
                                            return_dict=True)
            speech_logits = speech_output.logits
        '''
        if any(loss_name in speech_llm_related_losses for loss_name in self.loss_names):
            #print(f"in if any(loss_name in speech_llm_related_losses for loss_name in self.loss_names):")
            #print(f"lora")
            lora_audio_mask = torch.cat([torch.zeros_like(start_mask),
                                         torch.zeros_like(audio_instruction_mask),
                                         torch.ones_like(speech_input_mask),
                                         torch.zeros_like(suffix_mask)], dim=1)
            self.update_lora_mask(lora_audio_mask, False)
            speech_embeds_bf16 = speech_embeds.to(torch.bfloat16)
            speech_mask_bf16 = speech_mask.to(torch.bfloat16)
            speech_output_bf16 = self.qwen_model(inputs_embeds=speech_embeds_bf16, attention_mask=speech_mask_bf16,
                                            position_ids=speech_mask_bf16.cumsum(dim=-1) - 1, output_hidden_states=True,
                                            return_dict=True)
            speech_logits_bf16 = speech_output_bf16.logits
            speech_logits = speech_logits_bf16.to(torch.float32)

        total_loss = input_embeds.new_zeros(())
        input_er_loss = input_embeds.new_zeros(())
        input_er_w2v2_loss = input_embeds.new_zeros(())
        return_er_flag = False
        for loss_name in self.loss_names:
            if loss_name == "response_ce":
                shifted_logits = speech_logits[..., :-1, :].contiguous()
                shifted_labels = speech_labels[..., 1:].contiguous()
                #print(f"in if loss_name == response_ce:\nshifted_logits:{len(shifted_logits)}\n{shifted_logits}\nshifted_labels:{len(shifted_labels)}\n{shifted_labels}")
                loss = F.cross_entropy(shifted_logits[shifted_labels != -100],
                                       shifted_labels[shifted_labels != -100], reduction="mean")
                total_loss += loss
            elif loss_name == "response_kl":
                loss = F.kl_div(
                    F.log_softmax(speech_logits[speech_response_kd_labels == 1] / self.config.kd_temperature, dim=-1),
                    F.softmax(text_logits[text_response_kd_labels == 1] / self.config.kd_temperature, dim=-1),
                    reduction="batchmean"
                )
                total_loss += loss
            elif loss_name == "input_kl":
                check_shape(input_labels, speech_input_labels)
                loss = F.kl_div(
                    F.log_softmax(speech_logits[speech_kd_labels == 1] / self.config.kd_temperature, dim=-1),
                    F.softmax(text_logits[input_kd_labels == 1] / self.config.kd_temperature, dim=-1),
                    reduction="batchmean"
                )
                total_loss += loss
            elif loss_name == "cif":
                if speech_pred_num_tokens is None:
                    raise RuntimeError("predicted_num_tokens not set but cif_loss is requested")
                loss = F.l1_loss(speech_pred_num_tokens/input_mask.sum(-1), torch.ones_like(speech_pred_num_tokens),
                                  reduction="mean")
                total_loss += loss
                # loss_str += f"{loss_name}: {loss.item():.4f}, "
            elif loss_name == "input_er":
                #print("in elif loss_name == input_er")
                hidden_states = speech_input_embeds.clone()
                hidden_states[speech_input_mask == 0] = 0.0
                pooled_output = hidden_states.sum(dim=1) / speech_input_mask.sum(dim=1).view(-1, 1)
                er_logits = self.hidden2emotion(pooled_output)
                loss = F.cross_entropy(er_logits.view(-1, self.config.num_emotions), emotion_labels.view(-1))
                total_loss += loss
                input_er_loss += loss
                return_er_flag = True
            elif loss_name == "input_er_avd":
                hidden_states = speech_input_embeds.clone()
                hidden_states[speech_input_mask == 0] = 0.0
                pooled_output = hidden_states.sum(dim=1) / speech_input_mask.sum(dim=1).view(-1, 1)
                avd_logits = self.hidden2w2v2(pooled_output)
                avd_labels = torch.stack([arousal_ans, valence_ans, dominance_ans], dim=1)
                loss = F.mse_loss(avd_logits, avd_labels)
                #print(f"in 347 loss{loss} and avd label= {avd_labels}, pred_logit:{avd_logits}")
                #loss = F.cross_entropy(avd_logits.view(-1, self.config.num_emotions), emotion_labels.view(-1))
                total_loss += loss
                input_er_loss += loss
                return_er_flag = True
            else:
                raise RuntimeError(f"Unsupported loss name: {loss_name}")

            #elif loss_name == "input_er_w2v2":
                #print("in elif loss_name == input_er_w2v2")
                #hidden_states = emo_ad.clone()
                #hidden_states[emo_mask == 0] = 0.0
                #pooled_output = hidden_states.sum(dim=1) / emo_mask.sum(dim=1).view(-1, 1)
                #er_logits = self.w2v2_hidden2emotion(pooled_output)
                #loss = F.cross_entropy(er_logits.view(-1, self.config.num_emotions), emotion_labels.view(-1))
                #total_loss += loss
                #input_er_w2v2_loss += loss
                #return_w2v2_er_flag = True

        #print(f"stop{idd}")
        if return_er_flag is True:
            return {"loss": total_loss, "er_loss": input_er_loss}
        else:
            return {"loss": total_loss}
        #elif return_w2v2_er_flag is True:
        #    #print(f"in return_w2v2_er:\nloss{total_loss}\nw2v2_loss:{input_er_w2v2_loss}")
        #   return {"loss": total_loss, "w2v2_er_loss": input_er_w2v2_loss}



    def add_lora(self, lora_config, lora_scope="global"):
        if self.config.lora_config:
            logger.warning(f"add_lora ignored as model already has lora enabled")
        else:
            self.lora_config = lora_config
            self.config.lora_config = lora_config.to_dict()
            self.qwen_model = LoraModel(self.qwen_model, self.lora_config, "default")
            self.config.lora_scope = lora_scope

    def update_lora_mask(self, audio_mask, inference_mode: bool):
        if not self.config.lora_config or self.config.lora_scope == "global":
            return

        self.qwen_model.update_inference_mode(inference_mode)
        if self.config.lora_scope == "audio":
            self.qwen_model.update_lora_mask("default", audio_mask)
        elif self.config.lora_scope == "text":
            self.qwen_model.update_lora_mask("default", torch.ones_like(audio_mask) - audio_mask)
        elif self.config.lora_scope == "global":
            pass # do nonthing as official peft uses global lora
        else:
            raise ValueError(f"The scope value {self.config.lora_scope} for lora adapter 'default' is not supported")

    def merge_lora(self):
        if hasattr(self, 'lora_config'):
            if self.config.lora_scope != "global":
                raise ValueError(f"cannot call merge_lora when the lora_scope is not global ("
                                 f"{self.config.lora_scope})")
            self.qwen_model = self.qwen_model.merge_and_unload()
            self.config.lora_config = {}
            del self.lora_config
        else:
            raise ValueError("cannot call merge_lora when no self.lora_config is set")

    def get_speech_features(self, speech_values, speech_attention_mask, num_tokens=None):
        w2v_args = {
            "input_features": speech_values,
            "attention_mask": speech_attention_mask,
        }
        output = self.whisper_model(**w2v_args)
        speech_embeds = output.last_hidden_state # B x T x C
        attention_mask = length_to_attention_mask(output.output_lengths)

        # {len(num_tokens[0])}x
        #print(f"in get_speech_features before adapter:/nspeech_embeds:{len(speech_embeds)}x{len(speech_embeds[0])}x{len(speech_embeds[0][0])}/n{speech_embeds}/natt_mask:{len(attention_mask)}x{len(attention_mask[0])}/n{attention_mask}/nnum_token:{len(num_tokens)}/n{num_tokens}")

        speech_embeds, speech_atts, speech_logits, speech_cif_alphas, speech_pred_num_tokens = \
            self.adapter(speech_embeds, attention_mask, num_tokens)
        #print(f"in get_speech_features after adapter:/nspeech_embeds:{len(speech_embeds)}x{len(speech_embeds[0])}x{len(speech_embeds[0][0])}/n{speech_embeds}/natt_mask:{len(attention_mask)}x{len(attention_mask[0])}/n{attention_mask}/nnum_token:{len(num_tokens)}/n{num_tokens}")

        return speech_embeds, speech_atts, speech_logits, speech_cif_alphas, speech_pred_num_tokens

    def get_speech_features_addw2v2(self, speech_values, speech_attention_mask, emo_embed, num_tokens=None):
        w2v_args = {
            "input_features": speech_values,
            "attention_mask": speech_attention_mask,
        }
        output = self.whisper_model(**w2v_args)
        speech_embeds = output.last_hidden_state # B x T x C
        attention_mask = length_to_attention_mask(output.output_lengths)

        emo_append = emo_embed.unsqueeze(1)
        emo_append = emo_append.expand(-1,speech_embeds.size(1),-1)
        #print(f"in get_speech_features_addw2v2 before adapter:/nspeech_embeds:{len(speech_embeds)}x{len(speech_embeds[0])}x{len(speech_embeds[0][0])}/n{speech_embeds}/natt_mask:{len(attention_mask)}x{len(attention_mask[0])}/n{attention_mask}/nnum_token:{len(num_tokens)}/n{num_tokens}")
        #print(f"in get_speech_features_addw2v2 before adapter:\nemo_expand:{emo_append.shape}\n{emo_append}")
        speech_embeds_w2v2 = torch.cat((speech_embeds,emo_append),dim = 2)
        #print(f"in get_speech_features_addw2v2 after cat before adapter:\nspeech_embed_w2v2:{speech_embeds_w2v2.shape}\n{speech_embeds_w2v2}")
        #print(f"in get_speech_features_addw2v2 before adapter:attention mask:{attention_mask.shape}\n{attention_mask}")
        speech_embeds, speech_atts, speech_logits, speech_cif_alphas, speech_pred_num_tokens = \
            self.w2v2cat_adapter(speech_embeds_w2v2, attention_mask, num_tokens)
        #print(f"in get_speech_features_addw2v2 after adapter:/nspeech_embeds:{len(speech_embeds)}x{len(speech_embeds[0])}x{len(speech_embeds[0][0])}/n{speech_embeds}/natt_mask:{len(attention_mask)}x{len(attention_mask[0])}/n{attention_mask}/nnum_token:{len(num_tokens)}/n{num_tokens}")

        return speech_embeds, speech_atts, speech_logits, speech_cif_alphas, speech_pred_num_tokens

    def get_pure_emo_features(self, raw_speech, sampling_rate, num_tokens=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        aro_list = []
        dom_list = []
        val_list = []
        emb_list = []
        for sample in raw_speech:
            index = process_func_revised_resample(self.processor,self.w2v2,device,sample, sampling_rate = sampling_rate, embeddings=False)
            arousal = index[0][0]
            dominance = index[0][1]
            valence = index[0][2]
            emb = process_func_revised_resample(self.processor,self.w2v2,device,sample, sampling_rate = sampling_rate, embeddings=True)
            emo_embed = emb[0]
            aro_list.append(arousal)
            dom_list.append(dominance)
            val_list.append(valence)
            emb_list.append(emo_embed)
        aro_list = torch.FloatTensor(np.array(aro_list))
        dom_list = torch.FloatTensor(np.array(dom_list))
        val_list = torch.FloatTensor(np.array(val_list))
        emb_list = torch.FloatTensor(np.array(emb_list))

        attention_mask = emb_list.new_ones(emb_list.size(0), emb_list.size(1), dtype=torch.int64)
        # num_tokens=None >> use subsampler can be ignored

        emb_list = emb_list.to(device)
        return aro_list, dom_list, val_list, emb_list

    def get_emo_features(self, raw_speech, sampling_rate, num_tokens=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        aro_list = []
        dom_list = []
        val_list = []
        emb_list = []
        for sample in raw_speech:
            index = process_func_revised_resample(self.processor,self.w2v2,device,sample, sampling_rate = sampling_rate, embeddings=False)
            arousal = index[0][0]
            dominance = index[0][1]
            valence = index[0][2]
            emb = process_func_revised_resample(self.processor,self.w2v2,device,sample, sampling_rate = sampling_rate, embeddings=True)
            emo_embed = emb[0]
            aro_list.append(arousal)
            dom_list.append(dominance)
            val_list.append(valence)
            emb_list.append(emo_embed)
        aro_list = torch.FloatTensor(np.array(aro_list))
        dom_list = torch.FloatTensor(np.array(dom_list))
        val_list = torch.FloatTensor(np.array(val_list))
        emb_list = torch.FloatTensor(np.array(emb_list))

        attention_mask = emb_list.new_ones(emb_list.size(0), emb_list.size(1), dtype=torch.int64)
        # num_tokens=None >> use subsampler can be ignored

        emb_list = emb_list.to(device)
        #print(f"get_emo device {emb_list.device}")

        emb_list = emb_list.unsqueeze(1)
        emo = self.w2v2_adapter(emb_list)
        #emo_expand = emo.unsqueeze(1)
        #emo_expand = emb_list.unsqueeze(1)
        #emo_expand = emo_expand.repeat(1, 1, 4)
        return aro_list, dom_list, val_list, emb_list, emo #emo_expand

    def embedding_to_token_id2text(self, embedding):

        token_embeddings = self.qwen_model.get_input_embeddings().weight
        similarity_scores = F.cosine_similarity(embedding, token_embeddings, dim=-1)
        best_match_token_id = torch.argmax(similarity_scores).item()

        return best_match_token_id

    def load_w2v2embed(self,path):
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if "emo" in line:
                    emo_ans = line
                    if "sad" in emo_ans:
                        emo_label = self.emotion2idx["sad"]#"sad" #1
                    if "happy" in emo_ans:
                        emo_label = self.emotion2idx["happy"]#"happy" #2
                    if "neutral" in emo_ans:
                        emo_label = self.emotion2idx["neutral"]#"neutral" #3
                    if "angry" in emo_ans:
                        emo_label = self.emotion2idx["angry"]#"angry" #4
                    if "surprise" in emo_ans:
                        emo_label = self.emotion2idx["surprise"]#"surprise" #5
                else:
                    w2v2_list = json.loads(line)
                    w2v2_emb = torch.tensor(w2v2_list)
                    self.w2v2dict[emo_label] = w2v2_emb
        #print(f"print in line load_w2v2emb in modeling_blsp w2v2dict:{self.w2v2dict}")

    def get_w2v2emb_pretrained(self, emotion_labels):
        first = 0
        for emo in emotion_labels:
            emo_index = emo.item()
            new = self.w2v2dict[emo_index].unsqueeze(0)
            if first == 0:
                emb = new
                first = 1
            else:
                emb = torch.cat((emb, new), dim=0)
        #print(f"print in line get_w2v2emb in modeling_blsp\nemotion_labels:{emotion_labels}\nw2v2emb:{emb.shape}\n{emb}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        emb = emb.to(device)
        return emb

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        suffix_input_ids,
        suffix_attention_mask,
        raw_speech,
        sampling_rate,
        reference,
        w2v2_sep_ids,
        w2v2_sep_mask,
        speech_values=None,
        speech_attention_mask=None,
        generation_config=None,
        stop_words_ids=None,
    ):
        inputs_embeds, input_attention_mask, lora_audio_mask = [], [], []

        prefix_embeds = self.qwen_model.get_input_embeddings()(input_ids)
        inputs_embeds.append(prefix_embeds)
        input_attention_mask.append(attention_mask)
        lora_audio_mask.append(torch.zeros_like(attention_mask))

        '''
        w2v2_sep_emb = self.qwen_model.get_input_embeddings()(w2v2_sep_ids)
        inputs_embeds.append(w2v2_sep_emb)
        input_attention_mask.append(w2v2_sep_mask)
        lora_audio_mask.append(torch.zeros_like(w2v2_sep_mask))


        #emo_ad = self.qwen_model.get_input_embeddings()(w2v2_sep_ids)

        arousal, dominance, valence, emo_embed, emo_ad = \
            self.get_emo_features(raw_speech,sampling_rate)
        emo_ad = emo_ad*10
        emo_mask = torch.tensor((), dtype=torch.int64)
        emo_mask = emo_mask.new_ones(emo_ad.size(0), emo_ad.size(1),dtype=torch.int64)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        emo_mask = emo_mask.to(device)
        inputs_embeds.append(emo_ad)
        input_attention_mask.append(emo_mask)
        lora_audio_mask.append(torch.ones_like(emo_mask))


        print(f"emotional embedding:\n{emo_ad}\nmask:\n{emo_mask}")
        '''

        arousal, dominance, valence, emo_embed = self.get_pure_emo_features(raw_speech, sampling_rate)

        """
        #w2v2_embpath = "/home/jinghan/Desktop/blsp-emo/examples/test/extractw2v2_from_output_my_sess5_iemocap_w2v2cat_pv2_epo3_embedsave.jsonl"
        #self.load_w2v2embed(w2v2_embpath)
        #print(f"test in generate 593:{self.w2v2dict}")

        #Get direct w2v2

        emo_list = []
        for emo_str in reference:
            #print(f"reference:{emo_str} with {self.emotion2idx[emo_str]}")
            emo_list.append(self.emotion2idx[emo_str])
        emo_list = torch.tensor(emo_list)
        emo_embed = self.get_w2v2emb_pretrained(emo_list)
        """

        if speech_values is not None:
            speech_embeds, speech_attention_mask, _, _, _ = self.get_speech_features_addw2v2(speech_values, speech_attention_mask, emo_embed)
            #print(f"in Blsp2Model forward/line_153:\nspeech value:{len(speech_values)} x {len(speech_values[0])} x {len(speech_values[0][0])}\n{speech_values}\nspeech input embed:{len(speech_embeds)} x {len(speech_embeds[0])} x {len(speech_embeds[0][0])}\n{speech_embeds}\n")
            #print(f"speech embed:\n{speech_embeds}\nmask:\n{speech_attention_mask}")
            inputs_embeds.append(speech_embeds)
            input_attention_mask.append(speech_attention_mask)
            lora_audio_mask.append(torch.ones_like(speech_attention_mask))



        suffix_embeds = self.qwen_model.get_input_embeddings()(suffix_input_ids)
        inputs_embeds.append(suffix_embeds)
        input_attention_mask.append(suffix_attention_mask)
        lora_audio_mask.append(torch.zeros_like(suffix_attention_mask))

        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        input_attention_mask = torch.cat(input_attention_mask, dim=1)
        lora_audio_mask = torch.cat(lora_audio_mask, dim=1)

        self.update_lora_mask(lora_audio_mask, True)

        inputs_embeds_bf16 = inputs_embeds.to(torch.bfloat16)
        return self.qwen_model.generate(
                inputs_embeds=inputs_embeds_bf16,
                attention_mask=input_attention_mask,
                generation_config=generation_config,
                stop_words_ids=stop_words_ids
            )
        """
        return {"text_id":self.qwen_model.generate(
            inputs_embeds=inputs_embeds_bf16,
            attention_mask=input_attention_mask,
            generation_config=generation_config,
            stop_words_ids=stop_words_ids
        ), "w2v2emb_ids": None}#self.embedding_to_token_id2text(emo_ad)}
        """

    @torch.no_grad()
    def chat(
        self,
        history,
        generation_config,
        stop_words_ids,
        device,
    ):
        inputs_embeds = []
        lora_audio_mask = []

        for h in history:
            if len(h) == 1:
                ### text
                input_ids = h[0].to(device)
                embeds = self.qwen_model.get_input_embeddings()(input_ids)
                inputs_embeds.append(embeds)
                lora_audio_mask.append(torch.zeros_like(input_ids))
            elif len(h) == 2:
                ### speech
                speech_values, speech_attention_mask = h[0].to(device), h[1].to(device)
                speech_embeds, speech_attention_mask, _, _, _= self.get_speech_features(speech_values, speech_attention_mask)
                inputs_embeds.append(speech_embeds)
                lora_audio_mask.append(speech_attention_mask)
            else:
                raise NotImplementedError

        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        lora_audio_mask = torch.cat(lora_audio_mask, dim=1)
        self.update_lora_mask(lora_audio_mask, True)

        return self.qwen_model.generate(
            inputs_embeds=inputs_embeds,
            generation_config=generation_config,
            stop_words_ids=stop_words_ids
        )

    @torch.no_grad()
    def generate_embedsave(
        self,
        input_ids,
        attention_mask,
        suffix_input_ids,
        suffix_attention_mask,
        raw_speech,
        sampling_rate,
        w2v2_sep_ids,
        w2v2_sep_mask,
        speech_values=None,
        speech_attention_mask=None,
        generation_config=None,
        stop_words_ids=None,
    ):
        inputs_embeds, input_attention_mask, lora_audio_mask = [], [], []

        prefix_embeds = self.qwen_model.get_input_embeddings()(input_ids)
        inputs_embeds.append(prefix_embeds)
        input_attention_mask.append(attention_mask)
        lora_audio_mask.append(torch.zeros_like(attention_mask))

        '''
        w2v2_sep_emb = self.qwen_model.get_input_embeddings()(w2v2_sep_ids)
        inputs_embeds.append(w2v2_sep_emb)
        input_attention_mask.append(w2v2_sep_mask)
        lora_audio_mask.append(torch.zeros_like(w2v2_sep_mask))


        #emo_ad = self.qwen_model.get_input_embeddings()(w2v2_sep_ids)

        arousal, dominance, valence, emo_embed, emo_ad = \
            self.get_emo_features(raw_speech,sampling_rate)
        emo_ad = emo_ad*10
        emo_mask = torch.tensor((), dtype=torch.int64)
        emo_mask = emo_mask.new_ones(emo_ad.size(0), emo_ad.size(1),dtype=torch.int64)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        emo_mask = emo_mask.to(device)
        inputs_embeds.append(emo_ad)
        input_attention_mask.append(emo_mask)
        lora_audio_mask.append(torch.ones_like(emo_mask))


        print(f"emotional embedding:\n{emo_ad}\nmask:\n{emo_mask}")
        '''
        arousal, dominance, valence, emo_embed = self.get_pure_emo_features(raw_speech, sampling_rate)
        if speech_values is not None:
            speech_embeds, speech_attention_mask, _, _, _ = self.get_speech_features_addw2v2(speech_values, speech_attention_mask, emo_embed)
            #print(f"in Blsp2Model forward/line_153:\nspeech value:{len(speech_values)} x {len(speech_values[0])} x {len(speech_values[0][0])}\n{speech_values}\nspeech input embed:{len(speech_embeds)} x {len(speech_embeds[0])} x {len(speech_embeds[0][0])}\n{speech_embeds}\n")
            #print(f"speech embed:\n{speech_embeds}\nmask:\n{speech_attention_mask}")
            inputs_embeds.append(speech_embeds)
            input_attention_mask.append(speech_attention_mask)
            lora_audio_mask.append(torch.ones_like(speech_attention_mask))

        suffix_embeds = self.qwen_model.get_input_embeddings()(suffix_input_ids)
        inputs_embeds.append(suffix_embeds)
        input_attention_mask.append(suffix_attention_mask)
        lora_audio_mask.append(torch.zeros_like(suffix_attention_mask))

        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        input_attention_mask = torch.cat(input_attention_mask, dim=1)
        lora_audio_mask = torch.cat(lora_audio_mask, dim=1)

        self.update_lora_mask(lora_audio_mask, True)

        inputs_embeds_bf16 = inputs_embeds.to(torch.bfloat16)

        out = {}
        out['w2v2'] = emo_embed
        out['speech'] = speech_embeds

        return out
        '''
        return self.qwen_model.generate(
                inputs_embeds=inputs_embeds_bf16,
                attention_mask=input_attention_mask,
                generation_config=generation_config,
                stop_words_ids=stop_words_ids
            )
        '''
