import torch.nn as nn

from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
)
from transformers.models.electra.configuration_electra import ElectraConfig
from transformers.models.electra.modeling_electra import (
    ElectraPreTrainedModel,
    ElectraModel,
    ElectraClassificationHead,
    ElectraEmbeddings
)


class QuestionAnsweringForIntegratedElectra(ElectraPreTrainedModel):
    config_class = ElectraConfig
    base_model_prefix = "electra"

    def __init__(self, config=None):
        super().__init__(config)

        electra = ElectraModel(config)
        electra.init_weights()

        self.embedding_answer_span = ElectraEmbeddings(config)
        if config.embedding_size != config.hidden_size:
            self.embeddings_project_answer_span = nn.Linear(config.embedding_size, config.hidden_size)

        self.embedding_yesno = ElectraEmbeddings(config)
        if config.embedding_size != config.hidden_size:
            self.embeddings_project_yesno = nn.Linear(config.embedding_size, config.hidden_size)

        self.embedding_multi_choice = ElectraEmbeddings(config)
        if config.embedding_size != config.hidden_size:
            self.embeddings_project_multi_choice = nn.Linear(config.embedding_size, config.hidden_size)

        self._init_embedding_layers(config.name_or_path, config)

        self.encoder = electra.encoder
        # self.init_weights()
        self.num_labels = config.num_labels

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier_yesno = ElectraClassificationHead(config)
        self.classifier_multi_choice = ElectraClassificationHead(config)

    def forward(
        self,
        input_answer_span=None,
        input_yesno=None,
        input_multi_choice=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        assert input_answer_span is not None or input_yesno is not None or input_multi_choice is not None

        answer_span_output, yesno_output, multi_choice_output = None, None, None

        if input_answer_span:
            answer_span_output = self.forward_answer_span(**input_answer_span)
        if input_yesno:
            yesno_output = self.forward_yesno(**input_yesno)
        if input_multi_choice:
            multi_choice_output = self.forward_multi_choice(**input_multi_choice)

        return answer_span_output, yesno_output, multi_choice_output

    def forward_answer_span(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict, input_shape, device = self._get_tmp_setting_info(return_dict, input_ids, inputs_embeds)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        hidden_states = self.embedding_answer_span(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project_answer_span(hidden_states)

        discriminator_hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = discriminator_hidden_states[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (
                         start_logits,
                         end_logits,
                     ) + discriminator_hidden_states[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

    def forward_yesno(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict, input_shape, device = self._get_tmp_setting_info(return_dict, input_ids, inputs_embeds)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        hidden_states = self.embedding_yesno(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project_yesno(hidden_states)

        discriminator_hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier_yesno(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

    def forward_multi_choice(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict, input_shape, device = self._get_tmp_setting_info(return_dict, input_ids, inputs_embeds)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        hidden_states = self.embedding_multi_choice(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project_multi_choice(hidden_states)

        discriminator_hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier_multi_choice(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

    def _get_tmp_setting_info(self, return_dict, input_ids, inputs_embeds):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        return return_dict, input_shape, device

    def _init_embedding_layers(self, pre_trained_electra_path_name, config):
        embedding_param = {
            'answer_span': self.embedding_answer_span.state_dict(),
            'yesno': self.embedding_yesno.state_dict(),
            'multi_choice': self.embedding_multi_choice.state_dict()
        }

        model_electra = ElectraModel.from_pretrained(pre_trained_electra_path_name, config=config)
        model_electra_param = model_electra.state_dict()

        target_model_params = dict()
        data_type = ['answer_span', 'yesno', 'multi_choice']
        for target_key, target_value in model_electra_param.items():
            if 'embedding' in target_key:
                target_model_params[target_key] = target_value

        for d in data_type:
            for k, v in target_model_params.items():
                tmp_k = k[len('embeddings')+1:]
                embedding_param[d][tmp_k] = v

        self.embedding_answer_span.load_state_dict(embedding_param['answer_span'])
        self.embedding_yesno.load_state_dict(embedding_param['yesno'])
        self.embedding_multi_choice.load_state_dict(embedding_param['multi_choice'])










