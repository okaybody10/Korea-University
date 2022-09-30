# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import collections
from src import answer_span_evaluate, set_seed
import logging
import os
import torch
import timeit
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from processors.answer_span import AnswerSpanResult
from processors.yesno import YesNoResult
from processors.multi_choice_as_single import MultiChoiceResult
from src.utils import acc_score
from utils import (
    load_and_cache_examples_answer_span_family,
    load_and_cache_examples_yesno,
    load_and_cache_examples_multi_choice
)
from utils.data import multi_collate
from fastprogress.fastprogress import master_bar, progress_bar
from utils import compute_predictions_logits, compute_predictions_probs_yesno, compute_predictions_probs_multi_choice

logger = logging.getLogger(__name__)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def save_model(args, output_dir, model, tokenizer, optimizer, scheduler):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

    if args.save_optimizer:
        torch.save(optimizer, os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler, os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_dataloader = get_data_loader('train', train_dataset, args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * args.warmup_proportion), num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    model.zero_grad()
    mb = master_bar(range(int(args.num_train_epochs)))
    # Added here for reproductibility
    set_seed(args)

    total_train_loss = {
        'total': [],
        'answer_span': [],
        'yesno': [],
        'multi_choice': [],
    }
    _debug_batch_answer_span_list = {e: [] for e in range(args.num_train_epochs)}
    _debug_batch_yesno_list = {e: [] for e in range(args.num_train_epochs)}
    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch_answer_span, batch_yesno, batch_multi_choice = batch

            if batch_answer_span is not None:
                _debug_batch_answer_span_list[epoch].append(len(batch_answer_span[0]))
                batch_answer_span = tuple(t.to(args.device) for t in batch_answer_span)
                input_answer_span = {
                    "input_ids": batch_answer_span[0],
                    "attention_mask": batch_answer_span[1],
                    "token_type_ids": batch_answer_span[2],
                    "start_positions": batch_answer_span[3],
                    "end_positions": batch_answer_span[4]
                }
            else:
                input_answer_span = None

            if batch_yesno is not None:
                _debug_batch_yesno_list[epoch].append(len(batch_yesno[0]))
                batch_yesno = tuple(t.to(args.device) for t in batch_yesno)
                input_yesno = {
                    "input_ids": batch_yesno[0],
                    "attention_mask": batch_yesno[1],
                    "token_type_ids": batch_yesno[2],
                    "labels": batch_yesno[4]
                }
            else:
                input_yesno = None

            if batch_multi_choice is not None:
                batch_multi_choice = tuple(t.to(args.device) for t in batch_multi_choice)
                input_multi_choice = {
                    "input_ids": batch_multi_choice[0],
                    "attention_mask": batch_multi_choice[1],
                    "token_type_ids": batch_multi_choice[2],
                    "labels": batch_multi_choice[4]
                }
            else:
                input_multi_choice = None

            inputs = {
                'input_answer_span': input_answer_span,
                'input_yesno': input_yesno,
                'input_multi_choice': input_multi_choice
            }
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            answer_span_loss, yesno_loss, multi_choice_loss = outputs
            answer_span_loss = answer_span_loss[0] if answer_span_loss is not None else None
            yesno_loss = yesno_loss[0] if yesno_loss is not None else None
            multi_choice_loss = multi_choice_loss[0] if multi_choice_loss is not None else None
            losses = {
                'answer_span': answer_span_loss,
                'yesno': yesno_loss,
                'multi_choice': multi_choice_loss
            }

            tmp_loss = torch.zeros(len(losses))
            for i, (k, loss) in enumerate(losses.items()):
                if loss is None:
                    continue
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if len(args.cuda_visible_devices.split(",")) > 1:
                    loss = loss.mean()

                tmp_loss[i] = loss * args.loss_weight[k]
                if step % 10 == 0:
                    total_train_loss[k].append(tmp_loss[i].item())

            total_loss = sum(tmp_loss)
            if step % 10 == 0:
                total_train_loss['total'].append(total_loss.item())
            total_loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    save_model(args, output_dir, model, tokenizer, optimizer, scheduler)

            if 0 < args.max_steps < global_step:
                break

        if args.save_steps == 0:
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            save_model(args, output_dir, model, tokenizer, optimizer, scheduler)

        mb.write("Epoch {} done".format(epoch + 1))

        if 0 < args.max_steps < global_step:
            break

    return global_step, total_train_loss


def evaluate_answer_span(args, model, tokenizer, checkpoint, predict_file_answer_span, result_dir_name, result_name):

    dataset, examples, features = load_and_cache_examples_answer_span_family(args, tokenizer, phase='eval',
                                                                             target_file=predict_file_answer_span)

    global_step = str()
    eval_dataloader = get_data_loader('eval', dataset, args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(global_step))
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_answer_span': {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2]
                }
            }

            example_indices = batch[3]

            outputs = model(**inputs)[0]

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            logits = (outputs.start_logits, outputs.end_logits)
            logit = [to_list(logit[i]) for logit in logits]

            start_logits, end_logits = logit
            result = AnswerSpanResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    if not os.path.exists(result_dir_name):
        os.makedirs(result_dir_name)

    # Compute predictions
    output_prediction_file = os.path.join(result_dir_name, "predictions_{}.json".format(result_name))

    results = {}

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,   # output_prediction_file,
        None,   # output_nbest_file
        None,   # output_null_log_odds_file
        args.null_score_diff_threshold,
        tokenizer
    )

    # Compute the F1 and exact scores.
    prediction_texts = {}
    for p in predictions.keys():
        prediction_texts[p] = predictions[p]['predict_text']

    result = answer_span_evaluate(args.score_method_with_white_space, examples, prediction_texts)
    result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
    results.update(result)

    result_path = os.path.join(result_dir_name, 'result_' + result_name + ".txt")
    logger.info("***** Official Eval results *****")
    with open(result_path, "w", encoding='utf-8') as f:
        logger.info("****** %s ******", os.path.join(os.path.basename(result_dir_name), result_name))
        f.write("{}_answer_span\n".format(checkpoint))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            f.write(" {} = {}\n".format(key, str(results[key])))

    return_result = {
        'f1': result['f1'],
        'sample_num': result['total']
    }
    return return_result


def evaluate_yesno(args, model, tokenizer, predict_file_yesno, result_dir_name, result_name):

    dataset, examples, features = load_and_cache_examples_yesno(args, tokenizer, phase='eval',
                                                                target_file=predict_file_yesno)

    global_step = str()
    eval_dataloader = get_data_loader('eval', dataset, args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(global_step))
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_yesno': {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2]
                }
            }

            example_indices = batch[3]

            outputs = model(**inputs)[1]

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            logits = outputs[:1]
            logit = [to_list(logit[i]) for logit in logits][0]

            result = YesNoResult(unique_id, logit)

            all_results.append(result)

    if not os.path.exists(result_dir_name):
        os.makedirs(result_dir_name)

    output_prediction_file = os.path.join(result_dir_name, "predictions_{}.json".format(result_name))

    predictions = compute_predictions_probs_yesno(
        examples,
        features,
        all_results,
        output_prediction_file,
    )

    gold_answers = get_gold_answer_from_examples_yesno(examples)
    prediction_label = {}
    for p in predictions.keys():
        prediction_label[p] = predictions[p]['predict_label']
    accuracy = acc_score(gold_answers, prediction_label)

    result_path = os.path.join(result_dir_name, 'result_' + result_name + ".txt")
    with open(result_path, "w", encoding='utf-8') as f:
        logger.info("****** %s ******", os.path.join(os.path.basename(result_dir_name), result_name))
        logger.info("accuracy: {}".format(accuracy['acc'].item() * 100))
        f.write("accuracy: {}".format(accuracy['acc'].item() * 100))

    return_result = {
        'accuracy': accuracy['acc'].item() * 100,
        'sample_num': len(predictions)
    }

    return return_result


def evaluate_multi_choice(args, model, tokenizer, predict_file_multi_choice, result_dir_name, result_name):

    dataset, examples, features = load_and_cache_examples_multi_choice(args, tokenizer, phase='eval',
                                                                       target_file=predict_file_multi_choice)

    global_step = str()
    eval_dataloader = get_data_loader('eval', dataset, args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(global_step))
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_multi_choice': {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2]
                }
            }

            example_indices = batch[3]

            outputs = model(**inputs)[2]

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            logits = outputs[:1]
            logit = [to_list(logit[i]) for logit in logits][0]

            result = MultiChoiceResult(unique_id, logit)

            all_results.append(result)

    if not os.path.exists(result_dir_name):
        os.makedirs(result_dir_name)

    output_prediction_file = os.path.join(result_dir_name, "predictions_{}.json".format(result_name))

    predictions = compute_predictions_probs_multi_choice(
        examples,
        features,
        all_results,
        output_prediction_file,
    )

    final_preds = collections.OrderedDict()
    for example_id, probabilities in predictions.items():
        final_preds[example_id] = np.argmax(probabilities)
    gold_answers = get_gold_answer_from_examples_multi_choice(examples)
    prediction_label = {}
    for p in predictions.keys():
        prediction_label[p] = predictions[p]['predict_label']
    accuracy = acc_score(gold_answers, prediction_label)

    result_path = os.path.join(result_dir_name, 'result_' + result_name + ".txt")
    with open(result_path, "w", encoding='utf-8') as f:
        logger.info("****** %s ******", os.path.join(os.path.basename(result_dir_name), result_name))
        logger.info("accuracy: {}".format(accuracy['acc'].item() * 100))
        f.write("accuracy: {}".format(accuracy['acc'].item() * 100))

    return_result = {
        'accuracy': accuracy['acc'].item() * 100,
        'sample_num': len(predictions)
    }

    return return_result


def get_gold_answer_from_examples_yesno(examples):
    gold_answers = collections.OrderedDict()
    for example in examples:
        example_key = example.qas_id
        gold_answers[example_key] = 0 if example.label==False else 1

    return gold_answers


def get_gold_answer_from_examples_multi_choice(examples):
    gold_answers = collections.OrderedDict()
    for example in examples:
        example_key = example.qas_id
        if example.label == 1:
            gold_answers[example_key] = int(example.qas_id[-1])

    return gold_answers


def get_data_loader(phase, dataset, batch_size):
    if phase == 'train':
        train_sampler = RandomSampler(dataset)
        return DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=multi_collate)
    else:
        eval_sampler = SequentialSampler(dataset)
        return DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)


