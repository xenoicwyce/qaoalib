import numpy as np

def to_serializable(o):
    """
    Used in JSON.dump to convert some non-serializable numpy objects
    to serializable objects.

    Args:
        o (TYPE): Object to be converted.

    Returns:
        JSON-serializable object.

    """
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)

def cast_key(dict_, dtype=int):
    return {dtype(k): v for k, v in dict_.items()}
