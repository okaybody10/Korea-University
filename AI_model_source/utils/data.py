import torch
from collections import abc as container_abcs

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def multi_collate(batch):
    answer_span_batch = [data for data in batch if data[-1].item() == 0]
    yesno_batch = [data for data in batch if data[-1].item() == 1]
    multi_choice_batch = [data for data in batch if data[-1].item() == 2]

    answer_span_data, yesno_data, multi_choice_data = None, None, None
    if len(answer_span_batch) != 0:
        answer_span_data = stack_element(answer_span_batch)
    if len(yesno_batch) != 0:
        yesno_data = stack_element(yesno_batch)
    if len(multi_choice_batch) != 0:
        multi_choice_data = stack_element(multi_choice_batch)

    return answer_span_data, yesno_data, multi_choice_data


def stack_element(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)

    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [stack_element(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))










