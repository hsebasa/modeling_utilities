
import random
import string


def generate_random_variable_name(length: int = 8) -> str:
    """
    Generate a random valid Python variable name.

    Parameters
    ----------
    length : int, default=8
        The length of the random variable name.

    Returns
    -------
    str
        A random valid Python variable name.

    Notes
    -----
    - The name starts with a letter or underscore.
    - Subsequent characters can be letters, digits, or underscores.
    """
    if length < 1:
        raise ValueError("Length must be at least 1")

    first_char = random.choice(string.ascii_letters + "_")
    other_chars = ''.join(random.choices(string.ascii_letters + string.digits + "_", k=length - 1))
    return first_char + other_chars
