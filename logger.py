import datetime
import logging
import logging.handlers
import torch
import re
from shutil import copyfile
import sys
import h5py
import os
from tensorboardX import SummaryWriter
from sacred.observers import FileStorageObserver
from sacred.utils import create_basic_stream_logger
TENSORBOARD_LOGGER = None
FILE_LOGGER = None
EXPERIMENT = None


def print_warning(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_data_shape(data):
    if hasattr(data, 'shape'):
        dshape = data.shape
    elif type(data) == list or type(data) == tuple:
        dshape = (len(data),)
    else:
        dshape = (1,)

    return dshape


def var2num(x):
    return x.data.cpu().numpy()


class MemoryHandler(logging.handlers.MemoryHandler):
    def flush(self):
        self.acquire()
        try:
            for record in self.buffer:
                self.target.handle(record)
            self.buffer = []
            print('')
        finally:
            self.release()


class TensorboardLogger(object):
    def __init__(self, path, ex_info):
        self.writer = SummaryWriter(path)
        self.entries = {}
        ex_info["tensorflow"] = {}
        ex_info["tensorflow"]["logdirs"] = []
        path = os.path.abspath('tf_test')
        ex_info["tensorflow"]["logdirs"].append(path)

    def add_scalar(self, name, data):
        if name in self.entries:
            entry = self.entries[name]
            entry.update(data)
        else:
            entry = ScalarEntry(name, data)
            self.entries[name] = entry

        self.writer.add_scalar(entry.name, entry.to_value(), entry.n)

    def add_image(self, name, img, dataformat='CHW'):
        if name in self.entries:
            entry = self.entries[name]
            entry.update(img)
        else:
            entry = ImageEntry(name, img)
            self.entries[name] = entry

        self.writer.add_image(entry.name, entry.to_value(), entry.n, dataformats=dataformat)

    def add_images(self, name, imgs, dataformat='NCHW'):
        #TODO: check if it works
        if name in self.entries:
            entry = self.entries[name]
            entry.update(imgs)
        else:
            entry = ImageEntry(name, imgs)
            self.entries[name] = entry

        self.writer.add_images(entry.name, entry.to_value(), entry.n, dataformats=dataformat)

    def add_histogram(self, name, hist):
        raise NotImplementedError("add_histogram is not yet implemented")

    def add_figure(self, name, fig):
        raise NotImplementedError("add_figure is not yet implemented")

    def add_video(self, name, hist):
        raise NotImplementedError("add_video is not yet implemented")

    def add_audio(self, name, hist):
        raise NotImplementedError("add_audio is not yet implemented")

    def add_text(self, name, hist):
        raise NotImplementedError("add_text is not yet implemented")

    def add_graph(self, name, hist):
        raise NotImplementedError("add_graph is not yet implemented")

    def add_embedding(self, name, hist):
        raise NotImplementedError("add_embedding is not yet implemented")

    def add_pr_curve(self, name, hist):
        raise NotImplementedError("add_pr_curve is not yet implemented")

    def close(self):
        self.writer.close()


def sort_alpha_numeric(data):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def get_all_files_matching(base_name, path):
    reg_exp = re.sub("{.*}", "([0-9]+)", base_name)
    # TODO glob?
    files = os.listdir(path)
    reg_exp = re.compile(reg_exp)
    all_files = []
    for file in files:
        match = re.match(reg_exp, file)
        if match is None:
            continue
        all_files.append(os.path.join(path, file))
    return sort_alpha_numeric(all_files)


def get_last_file_number(base_name, path):
    """Gets the number of the last file
    Returns -1 if no file matching the base_name was found in path.
    """

    reg_exp = re.sub("{.*}", "([0-9]+)", base_name)
    # TODO glob?
    files = os.listdir(path)
    reg_exp = re.compile(reg_exp)
    max_value = -1
    for file in files:
        match = re.match(reg_exp, file)
        if match is None:
            continue
        value = int(match.group(1))
        if value > max_value:
            max_value = value

    return max_value


def get_new_numeric_name(base_name, path):
    """Finds all files with a given base name.
    
    Returns a filename counting upwards.

    base_name: contains curly brackets only once {}, place holder for a numeric value
    path: Directory where files are checked.
    """
    max_value = get_last_file_number(base_name, path)
    return base_name.format(max_value + 1)


# TODO 
# Converter function depending on where input is coming from
# Unified front , add_scalar, what about vectors?
# Log Level clean up, should be at every stage, but what if Backends add options
# possibility of merging file and console backends by logging level
# Scalar, running scalar, mean
# all have the save args capability
# separate Logging and summary
def create_path(path):
    if os.path.isdir(path):
        return path
    if os.path.isabs(path):
        # skip 'before' first slash
        path_folders = path.split(os.path.sep)[1:]
        base_path = os.path.sep
    else:
        path_folders = path.split(os.path.sep)
        base_path = ""

    for folder in path_folders:
        base_path = os.path.join(base_path, folder)
        if not os.path.isdir(base_path):
            os.mkdir(base_path)

    return path


def create_output(output_path, experiment):
    return os.path.join(output_path, experiment)


def report_after_batch(_run, logger, batch_id, batch_len, acc_time, max_mem, loss_mean):
    eta_seconds = acc_time * (batch_len / (batch_id + 1) - 1)
    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

    logger.info("Loss: %f (%d/%d)", loss_mean, batch_id + 1, batch_len)
    _run.info['batch_progress'] = "{:03f}% ({}/{})".format((batch_id + 1) / batch_len, batch_id + 1, batch_len)
    _run.info['eta'] = eta_string
    _run.info['speed'] = "{:.3f} it/s".format((batch_id + 1)/acc_time)
    _run.info['memory'] = max_mem / 1024.0 / 1024.0

def report_after_epoch(_run, epoch, max_epoch):
    _run.info['progress'] = "{:03f}% ({}/{})".format(
        epoch / max_epoch, epoch, max_epoch)

def report_after_training(_run, total_time, max_epoch):
    _run.info['total_training_time'] = str(datetime.timedelta(seconds=int(total_time)))
    _run.info['avg_training_time'] = str(datetime.timedelta(seconds=int(total_time / max_epoch)))

class Logger(object):

    model_name = "model_{}"

    @staticmethod
    def _make_model_name(iteration):
        return Logger.model_name.format(iteration)

    @staticmethod
    def get_all_model_paths(path):
        """args:
            path: path to experiment
        """
        if os.path.isdir(path):
            return get_all_files_matching(Logger.model_name, path)
        raise FileNotFoundError("Experiment dir could not be found: {}".format(path))

    @staticmethod
    def get_cfg_path(path):
        """args:
            path: path to experiment
        """
        if os.path.isdir(path):
            cfg_path = os.path.join(path, 'config.json')
            if os.path.isfile(cfg_path):
                return cfg_path

        raise FileNotFoundError("Config file could not be found for {}".format(path))

    def __init__(self, log_dir, origin="pytorch"):
        self.log_dir = log_dir
        create_path(self.log_dir)
        print("Logging to %s" % self.log_dir)

        self.origin = origin

    def close(self):
        pass

    def save_description(self, model):
        save_pytorch_description(model, self.log_dir)

    def get_model_path(self, iteration):
        path = os.path.join(self.log_dir, self._make_model_name(iteration))
        if not os.path.isfile(path):
            print("Checkpoint not found! {}".format(path))
            return None

        return path

    def get_log_dir(self):
        return self.log_dir

    def save_checkpoint(self, model, optimizer, epoch, model_cfg):
        path = os.path.join(self.log_dir, self._make_model_name(epoch))
        console_logger = get_logger()
        if os.path.isfile(path):
            console_logger.info("File already exists under %s", path)
            return None
        console_logger.info("Saving model under %s", path)
        checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_cfg': model_cfg
                }
        torch.save(checkpoint, path)
        return path

    def save_model_file(self, model_name, model_path="models"):
        """Saves the model file to the directory."""
        src = os.path.join(model_path, "{}.py".format(model_name))
        dst = os.path.join(self.log_dir, "{}.py".format(model_name))
        copyfile(src, dst)

import tempfile
class DummyLogger(Logger):
    def __init__(self):
        self._log_dir = None
        pass

    def write(self, name, data, dype=None):
        pass

    def add_scalar(self, name, data):
        pass

    def add_image(self, name, data, dataformat=''):
        pass

    def add_images(self, name, data, dataformat=''):
        pass
    
    def save_checkpoint(self, *args): 
        return None

    def close(self):
        pass

    @property
    def log_dir(self):
        if self._log_dir is None:
            self._log_dir = tempfile.mkdtemp()
        return self._log_dir


class H5Logger(Logger):

    DEFAULT_SIZE = 100
    LOG_FILE = "log_{}.h5"
    def __init__(self, *args):
        super().__init__(*args)
        log_file_name = get_new_numeric_name(self.LOG_FILE, self.log_dir)
        self.handle = h5py.File(os.path.join(self.log_dir, log_file_name), 'w')
        self.columns = {}

    def write(self, name, data, dtype=None):

        if name not in self.columns:
            if dtype == None:
                dtype = data.dtype
            dshape = get_data_shape(data)

            maxshape = tuple([None] + list(dshape))
            shape = tuple([self.DEFAULT_SIZE] + list(dshape))
            dataset = self.handle.create_dataset(name, shape=shape, maxshape=maxshape,
                                                 dtype=dtype)
            position = 0
            self.columns[name] = [dataset, position, self.DEFAULT_SIZE] # dataset handle, position, maxsize
        else:
            dataset, position, maxsize = self.columns[name]
            # position starts at 0
            if position + 1 > maxsize:
                # increase dataset by max_size
                maxsize = maxsize + self.DEFAULT_SIZE
                dshape = get_data_shape(data)
                shape = tuple([maxsize] + list(dshape))
                dataset.resize(shape)
                # print("Increased datset {} size to {}".format(name, maxsize))
                self.columns[name][2] = maxsize
        dataset[position] = data
        self.columns[name][1] += 1

    def close(self):
        self.handle.close()


class Entry(object):
    def __init__(self, name, every_n_steps=1):
        self.every_n_steps = every_n_steps
        self.name = name
        self.n = 0

    def update(self, *args):
        self.n += 1
        self._update(*args)

    def to_value(self):
        return self._to_value()


class ScalarEntry(Entry):
    def __init__(self, name, value, template="{}: {:.4f}"):
        super().__init__(name)
        self.value = value
        self.template = template

    def _update(self, value):
        self.value = value


    def _to_value(self):
        return self.value

    def __str__(self):
        if torch.Tensor is type(self.value):
            value = var2num(self.value)
        else:
            value = self.value
        return self.template.format(self.name, value)

class ImageEntry(Entry):
    def __init__(self, name, img):
        super().__init__(name)
        self.img = img

    def _update(self, img):
        self.img = img

    def _to_value(self):
        return self.img



def get_sacred_file_path(ex):
    """Returns files path of sacred file storage observer.

    Sacred run has to be already initialized!

    Returns: file path
    """
    if ex is None:
        return None
    for observer in ex.observers:
        if isinstance(observer, FileStorageObserver):
            return observer.dir
    return None


def build_logger(type, name, run):
    console_logger = get_logger()
    console_logger.info("creating logger %s", name)
    path = get_sacred_file_path(run)
    if type == "dummy":
        logger = DummyLogger()
    elif path is None:
        console_logger.info("Creating dummy logger. No path given.")
        logger= DummyLogger()
    elif type == "h5":
        console_logger.info("Logging to %s", path)
        logger = H5Logger(path)
    elif type == "tensorboard":
        logger = TensorboardLogger(path, run.experiment_info)
    else:
        raise NotImplemented("Type is not implemented: {}".format(type))

    return logger


def initialize(run):
    global FILE_LOGGER, TENSORBOARD_LOGGER, EXPERIMENT
    EXPERIMENT = run
    FILE_LOGGER = build_logger("h5", "h5", run)
    TENSORBOARD_LOGGER = build_logger("tensorboard", "tensorboard", run)


def get_logger(name=None):
    if EXPERIMENT is None:
        print("default logger")
        return logging.getLogger(name)
    elif EXPERIMENT.run_logger is None:
        EXPERIMENT.logger = create_basic_stream_logger()
    return EXPERIMENT.run_logger


def get_file_logger():
    global FILE_LOGGER
    if FILE_LOGGER is None:
        FILE_LOGGER = build_logger("dummy", "dummy", None)

    return FILE_LOGGER


def get_tensorboard_logger():
    global TENSORBOARD_LOGGER
    if TENSORBOARD_LOGGER is None:
        TENSORBOARD_LOGGER = build_logger("dummy", "dummy", None)
    return TENSORBOARD_LOGGER


def save_pytorch_description(model, path):
    path = os.path.join(path, "model.txt")
    with open(path, 'w') as f:
        print(model, file=f)

