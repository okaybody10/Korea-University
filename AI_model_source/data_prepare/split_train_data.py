import argparse
import json
import os
import random


def list_split(lst, n):
    each_length, m = divmod(len(lst), n)

    split_list = [lst[i * each_length + min(i, m):(i + 1) * each_length + min(i + 1, m)] for i in range(n)]

    return split_list


def get_split_data(origin_data, n):
    random.shuffle(origin_data['data'])
    data_list = origin_data['data']
    split_data_list = list(list_split(data_list, n))

    split_data = []
    for idx, current_data in enumerate(split_data_list):
        split_data.append({
            'version': '{}_{}'.format(origin_data['version'], str(idx)),
            'data': current_data
        })

    return split_data


def main(cli_args):
    data_dir = cli_args.data_dir
    target_data_filename = cli_args.target_file
    split_data_num = cli_args.split_num

    with open(os.path.join(data_dir, target_data_filename), 'r') as j:
        data = json.load(j)

    split_data = get_split_data(data, split_data_num)

    for each_data in split_data:
        filename = '{}.json'.format(each_data['version'])
        if os.path.exists(os.path.join(data_dir, filename)):
            continue

        print('Saving {}/{} ......'.format(data_dir, filename))
        with open(os.path.join(data_dir, filename), 'w', encoding='utf-8') as j:
            json.dump(each_data, j, indent='\t', ensure_ascii=False)


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--data_dir", type=str, required=True)
    cli_parser.add_argument("--target_file", type=str, required=True)
    cli_parser.add_argument("--split_num", type=int, required=True)

    cli_args = cli_parser.parse_args()
    main(cli_args)


