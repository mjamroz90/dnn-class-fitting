import os
import os.path as op
import json
import pickle


def load_files_from_dir(dir_name, ext):
    return [p for p in os.listdir(dir_name) if p.endswith(ext)]


def create_dir_if_not_exists(dir_name):
    if not op.exists(dir_name):
        os.makedirs(dir_name)


def load_celeb_files(celeb_faces_dir):
    return [op.join(celeb_faces_dir, p) for p in
            load_files_from_dir(celeb_faces_dir, ('jpg', 'png', 'jpeg'))]


def add_suffix_to_path(out_path, suffix):
    base_path = op.basename(out_path)
    dir_path = op.dirname(out_path)
    name, ext = base_path.split('.')[:-1], base_path.split('.')[-1]
    name = '.'.join(name)
    return op.join(dir_path, "%s_%s.%s" % (name, suffix, ext))


def read_json(file_name):
    with open(file_name) as f:
        return json.load(f)


def write_json(h, file_name):
    with open(file_name, 'w') as f:
        json.dump(h, f, indent=4)


def write_pickle(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


def read_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def chol_date_lib_path():
    import src.models

    models_path = op.abspath(op.dirname(src.models.__file__))
    lib_full_path = op.join(models_path, 'clustering/lib/tf_cholesky_date/build/libcholesky_date')

    if op.exists("%s.dylib" % lib_full_path):
        return "%s.dylib" % lib_full_path
    else:
        return "%s.so" % lib_full_path
