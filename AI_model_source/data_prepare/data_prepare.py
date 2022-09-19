from attrdict import AttrDict
import data_preparing_utils
import argparse
import json
import os


def main(args):
    with open(os.path.join('../config', args.config_file)) as f:
        config = AttrDict(json.load(f))

    base_data_dir = config.base_data_dir
    data_dir = config.data_dir
    save_dir = config.save_dir
    datatype = config.datatype_keywords
    if args.task == 'merge_base_data':
        target_datatype = list(datatype.values())
        data_preparing_utils.merge_base_data(base_data_dir, save_dir, target_datatype)

    elif args.task == 'split_train_val_test':
        target_datatype = list(datatype.values())
        data_preparing_utils.split_train_val_test(data_dir, save_dir, target_datatype)

    elif args.task == 'merge_answer_span_family':
        target_datatype_original = ['정답경계 추출형', 'Table 정답추출형', '절차(방법)', '응답 불가형']
        target_datatype = [datatype[target_type] for target_type in target_datatype_original]
        data_preparing_utils.merge_answer_span_family(data_dir, save_dir, target_datatype)

    elif args.task == 'shuffle_multi_choice_option':
        data_preparing_utils.shuffle_multi_choice_option(data_dir, save_dir)

    elif args.task == 'show_data_info':
        target_file_list = config.data_filename_list
        data_preparing_utils.show_data_info(data_dir, target_file_list)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    task_option = ['merge_base_data', 'split_train_val_test', 'merge_answer_span_family',
                   'shuffle_multi_choice_option', 'show_data_info']

    cli_parser.add_argument("--config_file", type=str, required=True)
    cli_parser.add_argument("--task", type=str, required=True, choices=task_option)
    cli_args = cli_parser.parse_args()
    main(cli_args)
