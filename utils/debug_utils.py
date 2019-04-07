import torch
import gc


def get_tracked_tensors():
    """
    https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/2
    :return: list of tracked tensors 
    """
    tensors = list()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj.data):
                tensors.append((type(obj), obj.size()))
        except:
            pass
    return tensors
