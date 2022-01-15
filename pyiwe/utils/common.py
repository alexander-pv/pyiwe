import datetime as dt


def timing(func: callable) -> callable:
    def wrapper(*args, **kwargs):
        t0 = dt.datetime.now()
        func(*args, **kwargs)
        t1 = dt.datetime.now()
        print('\nFinished. Seconds passed: %.4f\n' % ((t1 - t0).total_seconds()))

    return wrapper
