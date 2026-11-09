import os
import zipfile

# PyTorch 1.1 moves _download_url_to_file
# PyTorch 1.1 将 _download_url_to_file
#   from torch.utils.model_zoo to torch.hub
# 从 torch.utils.model_zoo 移到了 torch.hub。
# PyTorch 1.0 exists another _download_url_to_file
# PyTorch 1.0 中还有另一种 _download_url_to_file，
#   2 argument
# 接受两个参数。
# TODO: If you remove support PyTorch 1.0 or older,
# 待办：如果不再支持 PyTorch 1.0 及更早版本，
#       You should remove torch.utils.model_zoo
# 应移除 torch.utils.model_zoo，
#       Ref. PyTorch #18758
# 参考 PyTorch 问题 #18758。
#         https://github.com/pytorch/pytorch/pull/18758/commits
try:
    from torch.utils.model_zoo import _download_url_to_file
except ImportError:
    try:
        from torch.hub import download_url_to_file as _download_url_to_file
    except ImportError:
        from torch.hub import _download_url_to_file


def unzip(source_filename, dest_dir):
    with zipfile.ZipFile(source_filename) as zf:
        zf.extractall(path=dest_dir)


if __name__ == '__main__':
    _download_url_to_file('https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=1', 'saved_models.zip', None, True)
    unzip('saved_models.zip', '.')
