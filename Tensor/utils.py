

def isscalar(data):
    try:
        len(data)
        return False
    except TypeError:
        return True