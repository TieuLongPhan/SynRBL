import os
import csv
import json
import hashlib


def csv_reader(file):
    with open(file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            result_data = {}
            for k, v in zip(header, line):
                result_data[k] = v
            yield result_data


def json_reader(file):
    with open(file, "r") as f:
        json_data = json.load(f)
    if not isinstance(json_data, list):
        raise ValueError(
            "Top level json object sould be a list. "
            + r"Expected json structure: [{...},{...}]"
        )
    for json_entry in json_data:
        yield json_entry


class Dataset:
    def __init__(self, source):
        if isinstance(source, list):
            self.__data_reader = iter(source)
        elif isinstance(source, str):
            file_type = os.path.splitext(source)[1].replace(".", "").lower()
            if file_type == "csv":
                self.__data_reader = csv_reader(source)
            elif file_type == "json":
                self.__data_reader = json_reader(source)
            else:
                raise ValueError(
                    "File type '{}' is not supported as dataset source.".format(
                        file_type
                    )
                )
        else:
            raise ValueError(
                "'{}' is not a valid source for a dataset. "
                + "Use a file or list of data instead."
            )

    def __next__(self):
        return next(self.__data_reader)

    def __iter__(self):
        return self.__data_reader


class DataLoader:
    def __init__(self, data, batch_size=1):
        self.__data = data
        self.batch_size = batch_size
        self.__iter_stopped = False

    def __next__(self):
        if not self.__iter_stopped:
            return_data = []
            for _ in range(self.batch_size):
                try:
                    data_item = next(self.__data)
                    return_data.append(data_item)
                except StopIteration:
                    self.__iter_stopped = True
                    break
            return return_data
        else:
            raise StopIteration

    def __iter__(self):
        return self


class CacheManager:
    def __init__(self, cache_dir="./cache", cache_ext="cache"):
        self.__cache_dir = cache_dir
        self.__cache_ext = cache_ext
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.__cache_refs = {}
        for _, _, files in os.walk(cache_dir):
            for file in files:
                file_name, file_ext = os.path.splitext(file)
                cache_key = os.path.basename(file_name)
                file_path = os.path.join(os.path.abspath(cache_dir), file)
                if file_ext.replace(".", "") == cache_ext.lower():
                    self.__cache_refs[cache_key] = file_path

    def get_hash_key(self, data) -> str:
        dhash = hashlib.sha256()
        dhash.update(json.dumps(data, sort_keys=True).encode())
        return dhash.hexdigest()

    def is_cached(self, key) -> bool:
        return key in self.__cache_refs.keys()

    def load_cache(self, key):
        file = self.__cache_refs[key]
        with open(file, "r") as f:
            data = json.load(f)
        return data

    def write_cache(self, key, data):
        file = os.path.join(self.__cache_dir, "{}.{}".format(key, self.__cache_ext))
        with open(file, "w") as f:
            json.dump(data, f)
        return key
