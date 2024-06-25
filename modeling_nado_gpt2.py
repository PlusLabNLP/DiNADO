from modeling_gpt2_with_sdpa import GPT2LMHeadModel, GPT2Model, GPT2PreTrainedModel, Optional, Tuple, get_device_map, CausalLMOutputWithCrossAttentions, ModelOutput, dataclass, Union, CrossEntropyLoss
from transformers import GPT2Tokenizer
import torch
import torch.nn as nn

import numpy as np
def log1mexp(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    """
    mask = -np.log(2) < x  # x < 0
    x = torch.clamp_max(x, -eps)
    return torch.where(
        mask,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )

@dataclass
class CausalNADOOutputWithCrossAttentions(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `torch.FloatTensor` tuples of length `config.n_layers`, with each tuple containing the cached key,
            value states of the self-attention and the cross-attention layers if model is used in encoder-decoder
            setting. Only relevant if `config.is_decoder = True`.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    reg_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Union[Tuple[Tuple[torch.FloatTensor]], Tuple[Tuple[Tuple[torch.FloatTensor]]]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class GPT2DiNADOMergeLMHeadModel(GPT2LMHeadModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.norm_prediction_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.ReLU(),
            nn.Linear(config.n_embd * 4, 1, bias=False)
        )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        reference_model: Optional[GPT2LMHeadModel] = None,
    ) -> Union[Tuple, CausalNADOOutputWithCrossAttentions, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        reg_loss = None
        class_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits_policy = lm_logits

            shift_labels = input_ids[..., 1:].contiguous()

            r_policy = shift_logits_policy.log_softmax(dim=-1).clamp(-70., 0)
            if reference_model is not None:
                with torch.no_grad():
                    shift_logits_reference = reference_model(
                        input_ids,
                        past_key_values=None,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                hidden_states = shift_logits_reference.hidden_states[-1]
                betas = torch.nn.functional.logsigmoid(self.norm_prediction_head(hidden_states))
                shift_logits_reference = shift_logits_reference.logits
                # r_reference = loss_fct(shift_logits_reference.view(-1, shift_logits_reference.size(-1)), shift_labels.view(-1)).reshape_as(shift_labels)
                r_reference = shift_logits_reference.log_softmax(dim=-1)
                r = (r_policy - r_reference.clamp(-70., 0)).log_softmax(dim=-1)
                r = r - r.amax(dim=-1, keepdim=True)
                log_R = r + betas
                log_Ri = log_R[..., :-1, :].gather(dim=-1, index=shift_labels.unsqueeze(dim=-1)).reshape_as(shift_labels)
                log_1mRi = log1mexp(log_Ri)
                log_Ri_one_step_forward = (log_R + r_reference).logsumexp(dim=-1)[:, 1:]
                log_1mRi_one_step_forward = log1mexp(log_Ri_one_step_forward)
                Ri_one_step_forward = log_Ri_one_step_forward.exp()

                token_efft_mask = (shift_labels != self.config.eos_token_id).to(torch.long)
                token_efft_mask_last = torch.zeros_like(token_efft_mask)
                token_efft_mask_last[
                    torch.arange(token_efft_mask.size(0), device=token_efft_mask.device), token_efft_mask.sum(
                        dim=-1) - 1] = 1
                token_efft_mask[
                    torch.arange(token_efft_mask.size(0), device=token_efft_mask.device), token_efft_mask.sum(
                        dim=-1) - 1] = 0

                reg_loss = -Ri_one_step_forward * (log_Ri - log_Ri_one_step_forward) \
                           -(1. - Ri_one_step_forward) * (log1mexp(log_Ri) - log1mexp(log_Ri_one_step_forward))
                reg_loss = reg_loss * token_efft_mask

                class_loss = -labels.unsqueeze(dim=-1) * log_Ri - (1. - labels).unsqueeze(dim=-1) * log_1mRi

                class_loss = class_loss * token_efft_mask_last

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((class_loss, reg_loss, ) + output) if class_loss is not None else output

        if labels is None:
            return CausalLMOutputWithCrossAttentions(
                loss=None,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )
        else:
            return CausalNADOOutputWithCrossAttentions(
                loss=class_loss,
                reg_loss=reg_loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )

class GPT2DiNADOSoftLMHeadModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]
    def __init__(self, config, reference_model: GPT2LMHeadModel):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.r_prediction_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.norm_prediction_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.ReLU(),
            nn.Linear(config.n_embd * 4, 1, bias=False)
        )
        # Initialize weights and apply final processing

        self.post_init()
        self.r_prediction_head.weight.data.zero_()
        self._ref_model = [reference_model]

    @property
    def reference_model(self):
        if self._ref_model[0].device == self.device:
            self._ref_model[0].to(self.device)
        return self._ref_model[0]

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        return model_inputs

    @staticmethod
    def _reorder_cache_self(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )

    def _reorder_cache(self,
            past_key_values: Tuple[Tuple[Tuple[torch.Tensor]], Optional], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[Tuple[torch.Tensor]], Tuple[Tuple[torch.Tensor]]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return GPT2DiNADOSoftLMHeadModel._reorder_cache_self(past_key_values[0], beam_idx), self.reference_model._reorder_cache_self(past_key_values[1], beam_idx),



    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, CausalNADOOutputWithCrossAttentions, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_key_values is not None:
            past_key_values, past_key_values_base = past_key_values
        else:
            past_key_values_base = None

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        r = self.r_prediction_head(hidden_states).log_softmax(dim=-1).clamp(min=-70., max=0.)

        with torch.no_grad():
            base_model_output = self.reference_model(
                input_ids,
                past_key_values=past_key_values_base,
                attention_mask=attention_mask,
            )

        lm_logits = r + base_model_output.logits

        reg_loss = None
        class_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits_policy = lm_logits

            shift_labels = input_ids[..., 1:].contiguous()
            betas = torch.nn.functional.logsigmoid(self.norm_prediction_head(hidden_states))

            p_reference = base_model_output.logits.log_softmax(dim=-1)

            r = r - r.amax(dim=-1, keepdim=True)
            log_R = r + betas
            log_Ri = log_R[..., :-1, :].gather(dim=-1, index=shift_labels.unsqueeze(dim=-1)).reshape_as(shift_labels)
            log_1mRi = log1mexp(log_Ri)
            log_Ri_one_step_forward = (log_R + p_reference).logsumexp(dim=-1)[:, 1:]
            log_1mRi_one_step_forward = log1mexp(log_Ri_one_step_forward)
            Ri_one_step_forward = log_Ri_one_step_forward.exp()

            token_efft_mask = (shift_labels != self.config.eos_token_id).to(torch.long)
            token_efft_mask_last = torch.zeros_like(token_efft_mask)
            token_efft_mask_last[
                torch.arange(token_efft_mask.size(0), device=token_efft_mask.device), token_efft_mask.sum(
                    dim=-1) - 1] = 1
            token_efft_mask[
                torch.arange(token_efft_mask.size(0), device=token_efft_mask.device), token_efft_mask.sum(
                    dim=-1) - 1] = 0

            reg_loss = -Ri_one_step_forward * (log_Ri - log_Ri_one_step_forward) \
                       -(1. - Ri_one_step_forward) * (log1mexp(log_Ri) - log1mexp(log_Ri_one_step_forward))
            reg_loss = reg_loss * token_efft_mask

            class_loss = -labels.unsqueeze(dim=-1) * log_Ri - (1. - labels).unsqueeze(dim=-1) * log_1mRi

            class_loss = class_loss * token_efft_mask_last

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((class_loss, reg_loss, ) + output) if class_loss is not None else output

        if labels is None:
            return CausalNADOOutputWithCrossAttentions(
                loss=None,
                reg_loss=None,
                logits=lm_logits,
                past_key_values=(transformer_outputs.past_key_values, base_model_output.past_key_values),
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )
        else:
            return CausalNADOOutputWithCrossAttentions(
                loss=class_loss,
                reg_loss=reg_loss,
                logits=lm_logits,
                past_key_values=(transformer_outputs.past_key_values, base_model_output.past_key_values),
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )
