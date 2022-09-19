import logging
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
import os
import glob
from processors.answer_span import AnswerSpanProcessor, convert_examples_to_features_answer_span
from processors.yesno import YesNoProcessor, convert_examples_to_features_yesno
from processors.multi_choice_as_single import (
    MultiChoiceProcessor,
    convert_examples_to_features_multi_choice
)
from model.electra_integrated_qa_model import QuestionAnsweringForIntegratedElectra
from src import CONFIG_CLASSES

logger = logging.getLogger(__name__)


def load_model(args, phase):
    assert phase == 'train' or phase == 'eval'
    if phase == 'train':
        model_name_or_path = args.model_name_or_path

        logger.info(" model_name_or_path = %s", model_name_or_path)
        config = CONFIG_CLASSES[args.model_type].from_pretrained(
            args.model_name_or_path,
        )
        model = QuestionAnsweringForIntegratedElectra.from_pretrained(
            model_name_or_path,
            config=config
        )

        if args.device == "cuda" and len(args.cuda_visible_devices.split(",")) > 1:
            model = nn.DataParallel(model)
        model.to(args.device)

        return model
    else:
        checkpoints = list(
            os.path.dirname(c)
            for c in sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True))
        )

        check_dict = {int(checkpoint.split('checkpoint-')[-1]): i for i, checkpoint in enumerate(checkpoints)}
        sorted_checkpoints = sorted(check_dict)
        sorted_checkpoints_paths = [checkpoints[check_dict[checkpoint]] for checkpoint in sorted_checkpoints]

        # get target checkpoint
        target_index = args.target_epoch
        try:
            checkpoints = list()
            if type(target_index) == list or type(target_index) == tuple:
                target_index_list = target_index
            else:
                target_index_list = [target_index]

            for target_index in sorted(target_index_list):
                if target_index == -1:
                    prefix_i = len(sorted_checkpoints) - 1
                else:
                    prefix_i = target_index
                checkpoints.append((prefix_i, sorted_checkpoints_paths[target_index]))
        except IndexError:
            logging.info("Error]Loading checkpoint - %s", str(sorted_checkpoints_paths))
            exit(-1)

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        return checkpoints


def load_and_cache_examples_all(args, tokenizer, phase='train'):
    task = ['answer_span', 'yesno', 'multi_choice']
    dataset_filename_list_dict = {
        'answer_span': args.train_file_answer_span_family,
        'yesno': args.train_file_yesno,
        'multi_choice': args.train_file_multi_choice
    }
    preprocessing_functions = {
        'answer_span': load_and_cache_examples_answer_span_family,
        'yesno': load_and_cache_examples_yesno,
        'multi_choice': load_and_cache_examples_multi_choice
    }
    dataset_dict = dict()

    for t in task:
        cur_dataset_list = []
        for task_data_filename in dataset_filename_list_dict[t]:
            cur_dataset = preprocessing_functions[t](args, tokenizer, phase=phase, target_file=task_data_filename)
            cur_dataset_list.append(cur_dataset)
        cur_dataset_parameter_num = len(cur_dataset_list[0][:])
        cur_dataset_parameters = []
        for i in range(cur_dataset_parameter_num):
            cur_dataset_parameters.append(torch.cat([train_dataset[:][i] for train_dataset in cur_dataset_list]))

        dataset_dict[t] = TensorDataset(*cur_dataset_parameters)

    return dataset_dict


def load_and_cache_examples_answer_span_family(args, tokenizer, phase, target_file):
    cached_features_file_path = get_cached_features_file_path(args, 'answer_span', target_file)

    # Init features and dataset from cache if it exists
    if phase == 'train':
        if os.path.exists(cached_features_file_path):
            logger.info("Loading features from cached file %s", cached_features_file_path)
            dataset = torch.load(cached_features_file_path)['dataset']

        else:
            logger.info("Creating features from dataset file at %s %s", args.data_dir_answer_span_family, target_file)
            processor = AnswerSpanProcessor()

            examples = processor.get_train_examples(os.path.join(args.data_dir_answer_span_family),
                                                    filename=target_file)

            _, dataset = convert_examples_to_features_answer_span(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True,
                threads=args.threads,
            )

            logger.info("Saving features into cached file %s", cached_features_file_path)
            torch.save({"dataset": dataset}, cached_features_file_path)

        return dataset
    else:
        if os.path.exists(cached_features_file_path):
            logger.info("Loading features from cached file %s", cached_features_file_path)
            features_and_dataset = torch.load(cached_features_file_path)
            features, dataset, examples = (
                features_and_dataset["features"],
                features_and_dataset["dataset"],
                features_and_dataset["examples"],
            )
        else:
            logger.info("Creating features from dataset file at %s %s", args.data_dir_answer_span_family, target_file)
            processor = AnswerSpanProcessor()

            examples = processor.get_dev_examples(os.path.join(args.data_dir_answer_span_family),
                                                  filename=target_file)

            features, dataset = convert_examples_to_features_answer_span(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False,
                threads=args.threads,
            )

            logger.info("Saving features into cached file %s", cached_features_file_path)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file_path)
        return dataset, examples, features


def load_and_cache_examples_yesno(args, tokenizer, phase, target_file):
    cached_features_file_path = get_cached_features_file_path(args, 'yesno', target_file)

    if phase == 'train':
        if os.path.exists(cached_features_file_path):
            logger.info("Loading features from cached file %s", cached_features_file_path)
            dataset = torch.load(cached_features_file_path)['dataset']
        else:
            logger.info("Creating features from dataset file at %s %s", args.data_dir_yesno, target_file)
            processor = YesNoProcessor()

            examples = processor.get_examples(os.path.join(args.data_dir_yesno), filename=target_file)

            _, dataset = convert_examples_to_features_yesno(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True,
                threads=args.threads
            )

            logger.info("Saving features into cached file %s", cached_features_file_path)
            torch.save({"dataset": dataset}, cached_features_file_path)
        return dataset
    else:
        if os.path.exists(cached_features_file_path):
            logger.info("Loading features from cached file %s", cached_features_file_path)
            features_and_dataset = torch.load(cached_features_file_path)
            features, dataset, examples = (
                features_and_dataset["features"],
                features_and_dataset["dataset"],
                features_and_dataset["examples"],
            )
        else:
            logger.info("Creating features from dataset file at %s %s", args.data_dir_yesno, target_file)
            processor = YesNoProcessor()

            examples = processor.get_examples(os.path.join(args.data_dir_yesno), filename=target_file)

            features, dataset = convert_examples_to_features_yesno(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False,
                threads=args.threads
            )

            logger.info("Saving features into cached file %s", cached_features_file_path)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file_path)

        return dataset, examples, features


def load_and_cache_examples_multi_choice(args, tokenizer, phase, target_file):
    cached_features_file_path = get_cached_features_file_path(args, 'multi_choice', target_file)

    if phase == 'train':
        if os.path.exists(cached_features_file_path):
            logger.info("Loading features from cached file %s", cached_features_file_path)
            dataset = torch.load(cached_features_file_path)['dataset']
        else:
            logger.info("Creating features from dataset file at %s %s", args.data_dir_multi_choice, target_file)
            processor = MultiChoiceProcessor()

            examples = processor.get_examples(os.path.join(args.data_dir_multi_choice), filename=target_file)

            features, dataset = convert_examples_to_features_multi_choice(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                max_option_length=args.max_option_length,
                is_training=True,
                threads=args.threads
            )

            logger.info("Saving features into cached file %s", cached_features_file_path)
            torch.save({"dataset": dataset}, cached_features_file_path)
        return dataset
    else:
        if os.path.exists(cached_features_file_path):
            logger.info("Loading features from cached file %s", cached_features_file_path)
            features_and_dataset = torch.load(cached_features_file_path)
            features, dataset, examples = (
                features_and_dataset["features"],
                features_and_dataset["dataset"],
                features_and_dataset["examples"],
            )
        else:
            logger.info("Creating features from dataset file at %s %s", args.data_dir_multi_choice, target_file)
            processor = MultiChoiceProcessor()

            examples = processor.get_examples(os.path.join(args.data_dir_multi_choice), filename=target_file)

            features, dataset = convert_examples_to_features_multi_choice(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                max_option_length=args.max_option_length,
                is_training=False,
                threads=args.threads
            )

            logger.info("Saving features into cached file %s", cached_features_file_path)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file_path)

        return dataset, examples, features


def get_cached_features_file_path(args, task, target_file):
    # Load data features from cache or dataset file
    if task == 'answer_span':
        input_dir = args.data_dir_answer_span_family
    elif task == 'yesno':
        input_dir = args.data_dir_yesno
    else:   # multi_span
        input_dir = args.data_dir_multi_choice

    print("\n\n\ttarget is -", target_file, "\n")

    if not os.path.exists(os.path.join(input_dir, args.data_cache_dir)):
        os.makedirs(os.path.join(input_dir, args.data_cache_dir))

    cached_features_file_path = os.path.join(
        input_dir,
        args.data_cache_dir,
        "cached_{}_{}_inte".format(
            target_file.split(".json")[0],
            str(args.max_seq_length)
        )
    )

    return cached_features_file_path



