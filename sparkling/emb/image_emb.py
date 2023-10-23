import io
import os
import re
from abc import ABC, abstractmethod

from pyspark import SparkContext

from .deep_emb import DeepEmb


def _make_local_builder():
    def local_builder():
        import pyarrow.fs as fs
        return fs.LocalFileSystem()
    return local_builder


def _make_hdfs_builder(root_fs):
    def cmd_exec(cmd):
        from subprocess import Popen, PIPE
        return Popen(cmd, stdout=PIPE).stdout.read().rstrip().decode('utf-8')

    def find_libhdfs():
        cmd_exec('updatedb')
        lib_hdfs_path = cmd_exec(['locate', '-l', '1', 'libhdfs.so'])
        if len(lib_hdfs_path) == 0:
            raise ValueError('Failed to find libhdfs.so')
        return os.path.dirname(lib_hdfs_path)

    def hdfs_builder():
        import pyarrow.fs as fs
        if 'ARROW_LIBHDFS_DIR' not in os.environ:
            os.environ['ARROW_LIBHDFS_DIR'] = find_libhdfs()
            hadoop_cp = cmd_exec(['/usr/bin/hdfs', 'classpath', '--glob'])
            if 'CLASSPATH' in os.environ:
                os.environ['CLASSPATH'] = os.environ['CLASSPATH'] + ':' + hadoop_cp
            else:
                os.environ['CLASSPATH'] = hadoop_cp
        return fs.HadoopFileSystem.from_uri(root_fs)

    return hdfs_builder


def _resolve_fs(sc: SparkContext):
    standalone = re.compile('local(-cluster)?\\[.*]')
    if standalone.match(sc.master):
        return _make_local_builder()
    elif 'yarn' == sc.master:
        root_fs = sc._jsc.hadoopConfiguration().get('fs.defaultFS')
        if root_fs is None or len(root_fs) == 0:
            raise ValueError("Hadoop property 'fs.defaultFS' is not set")
        return _make_hdfs_builder(root_fs)
    raise ValueError(f"Can not automatically setup for 'f{sc.master}'")


class ImageBatch:
    """ Lightweight custom implementation for image reading and batch sampling """

    def __init__(self, series, fs, root_path, batch_size, processor, tensors):
        self.series, self.batch_size = series, batch_size
        self.fs, self.root_path = fs, root_path
        self.processor, self.tensors = processor, tensors
        self.cur_idx, self.len = 0, len(series)

    def _open_pil(self, path):
        from PIL import Image
        with self.fs.open_input_stream(self.root_path + path) as fstream:
            img_data = fstream.readall()
            return Image.open(io.BytesIO(img_data))

    def __iter__(self):
        return self

    def __next__(self):
        lower, upper = self.cur_idx, min(self.cur_idx + self.batch_size, self.len)
        if lower == upper:
            self.cur_idx = 0
            raise StopIteration

        pil_images = map(self._open_pil, self.series[lower:upper])
        self.cur_idx = upper
        return self.processor(list(pil_images), return_tensors=self.tensors)


class ImageEmb(DeepEmb, ABC):
    """
    Basic marker for image modality preprocessors.
    Also configures environment on each worker for correct communication with file storage
    """

    def __init__(self, name: str, model_name: str, orig_dim: int, reduce_dim: bool, **kwargs):
        fs_builder = _resolve_fs(SparkContext.getOrCreate())
        super().__init__(name, model_name, orig_dim, reduce_dim, fs_builder=fs_builder, **kwargs)

    @staticmethod
    @abstractmethod
    def from_zoo(name: str, reduce_dim: bool, model, root_path: str, batch_size: int):
        pass
