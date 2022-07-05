def assertion(condition, mess=""):
    try:
        assert condition, mess
    except AssertionError as e:
        raise e
        exit(-1)
