def assertion(condition, mess=""):
    """If condition is False, raises an AssertionError with message and exits a program.
    
    :param condition: condition to evaluate
    :param str mess: message that is raised by an AssertionError
    """

    try:
        assert condition, mess
    except AssertionError as e:
        raise e
        exit(-1)
