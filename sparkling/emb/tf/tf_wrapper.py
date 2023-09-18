import os
import tempfile
import zipfile


def _write_recursively(zipfile_to_save, system_path, zip_path):
    import tensorflow as tf
    if not tf.io.gfile.isdir(system_path):
        zipfile_to_save.write(system_path, zip_path)
    else:
        for file_name in tf.io.gfile.listdir(system_path):
            system_file_path = tf.io.gfile.join(system_path, file_name)
            zip_file_path = tf.io.gfile.join(zip_path, file_name)
            _write_recursively(zipfile_to_save, system_file_path, zip_file_path)


class TFWrapper:
    """ Custom serializer for hugging face's tensorflow models """

    def __init__(self, model):
        self.model = model

    def __getstate__(self):
        import tensorflow as tf

        temp_dir = tempfile.mkdtemp()
        try:
            model_path = os.path.join(temp_dir, 'model')
            self.model.save_pretrained(model_path)
            zip_path = os.path.join(temp_dir, 'model.keras')
            with zipfile.ZipFile(zip_path, 'w') as zipfile_to_save:
                _write_recursively(zipfile_to_save, model_path, '')
            with open(zip_path, 'rb') as f:
                data = f.read()
        except Exception as e:
            raise e
        else:
            return data
        finally:
            tf.io.gfile.rmtree(temp_dir)

    def __setstate__(self, state):
        import tensorflow as tf
        from transformers import TFAutoModel

        temp_dir = tempfile.mkdtemp()
        try:
            zip_path = os.path.join(temp_dir, 'model.keras')
            with open(zip_path, 'wb') as f:
                f.write(state)
            model_path = os.path.join(temp_dir, 'model')
            with zipfile.ZipFile(zip_path, 'r') as zipfile_to_load:
                zipfile_to_load.extractall(model_path)
            model = TFAutoModel.from_pretrained(model_path)
        except Exception as e:
            raise e
        else:
            self.model = model
        finally:
            tf.io.gfile.rmtree(temp_dir)
