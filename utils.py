from typing import Any, Dict, Tuple

import json
import pickle



def save_pickle(object: Any, filename: str) -> None:
    """
    Pickle the given object.

    :param object: Any serializable object to be pickled.
    :param filename: Filename to save the pickled object as.
    """
    with open(filename, "wb") as f:
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_json(dict: Dict, filename: str) -> None:
    """
    Save the given dictionary into a json file.

    :param dict: Dictionary to be saved as json.
    :param filename: Filename to save the dictionary as.
    """
    with open(filename, "w") as f:
        json.dump(dict, f)


def read_json(filename: str) -> Dict:
    """
    Read a json file.

    :param filename: Name of the json file.
    """
    with open(filename) as f: data = json.load(f)
    return data