

def is_not_none(data):
    return data is not None

def check_keys(dict1, dict2):
  return all(key in dict1 for key in dict2)