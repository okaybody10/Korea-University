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

import matplotlib.pyplot as plt
import argparse
import json

import logging
import os
import torch
from attrdict import AttrDict
from train_eval import train, evaluate_answer_span, evaluate_yesno, evaluate_multi_choice
from torch.utils.data.dataset import ConcatDataset
from src import (
    TOKENIZER_CLASSES,
    init_logger,
    set_seed,
)
import torch.nn as nn
from model.electra_integrated_qa_model import QuestionAnsweringForIntegratedElectra
from utils import load_model, load_and_cache_examples_all

logger = logging.getLogger(__name__)


def main(cli_args):
    # Read from config file and make args
    with open(os.path.join(cli_args.config_dir, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if type(args.output_dir) == str:
        args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    init_logger()
    set_seed(args)

    logging.getLogger("transformers.data.metrics.squad_metrics").setLevel(logging.WARN)  # Reduce model loading logs

    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
    )

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        model = load_model(args, phase='train')

        # train_dataset = load_and_cache_examples_all(args, tokenizer, evaluate=False, output_examples=False)
        train_dataset = load_and_cache_examples_all(args, tokenizer, phase='train')
        concat_train_dataset = ConcatDataset([train_dataset['answer_span'], train_dataset['yesno'],
                                              train_dataset['multi_choice']])

        global_step, total_train_loss = train(args, concat_train_dataset, model, tokenizer)
        plt.figure(1)
        plt.plot(total_train_loss['total'])
        plt.title('total loss sum')

        plt.figure(2)
        plt.plot(total_train_loss['answer_span'])
        plt.title('answer_span loss sum')

        plt.figure(3)
        plt.plot(total_train_loss['yesno'])
        plt.title('yesno loss sum')

        plt.figure(4)
        plt.plot(total_train_loss['multi_choice'])
        plt.title('multi_choice loss sum')

        plt.show()
    if args.do_eval:
        result_scores = {}
        checkpoints = load_model(args, phase='eval')
        checkpoint = checkpoints[0][1]
        result_dir_name = args.result_dir_name

        # Reload the model
        model = QuestionAnsweringForIntegratedElectra.from_pretrained(checkpoint)

        if args.device == "cuda" and len(args.cuda_visible_devices.split(",")) > 1:
            model = nn.DataParallel(model)
        model.to(args.device)

        for answer_span_predict_file in args.predict_file_answer_span_family:
            result_name = answer_span_predict_file.split('.json')[0]
            answer_span_result = evaluate_answer_span(args, model, tokenizer, checkpoint, answer_span_predict_file,
                                                      result_dir_name, result_name=result_name)
            result_scores[answer_span_predict_file] = answer_span_result

        for yesno_predict_file in args.predict_file_yesno:
            result_name = yesno_predict_file.split('.json')[0]
            yesno_result = evaluate_yesno(args, model, tokenizer, yesno_predict_file,
                                          result_dir_name, result_name=result_name)
            result_scores[yesno_predict_file] = yesno_result

        for multi_choice_predict_file in args.predict_file_multi_choice:
            result_name = multi_choice_predict_file.split('.json')[0]
            multi_choice_result = evaluate_multi_choice(args, model, tokenizer, multi_choice_predict_file,
                                                        result_dir_name, result_name=result_name)
            result_scores[multi_choice_predict_file] = multi_choice_result

        tmp_sum_sample_num_weighted_score = 0
        all_sample_num = 0
        print()
        logger.info("{} SCORES {}".format('='*25, '='*25))
        for k, v in result_scores.items():
            all_sample_num += v['sample_num']
            logger.info("filename: {}".format(k))
            logger.info("# of samples: {}".format(v['sample_num']))
            try:
                logger.info("F1 score: {:.2f}".format(v['f1']))
                tmp_sum_sample_num_weighted_score += v['f1']*v['sample_num']
            except KeyError:
                logger.info("Accuracy: {:.2f}".format(v['accuracy']))
                tmp_sum_sample_num_weighted_score += v['accuracy'] * v['sample_num']
            logger.info("-"*60)

        print()
        logger.info("=" * 24)
        logger.info("|| final score: {:.2f} ||".format(tmp_sum_sample_num_weighted_score/all_sample_num))
        logger.info("=" * 24)


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, required=True)

    cli_args = cli_parser.parse_args()

    main(cli_args)
