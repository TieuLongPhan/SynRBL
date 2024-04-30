import multiprocessing
import multiprocessing.pool
import functools

# workaround from https://github.com/joblib/joblib/pull/366
# remove when joblib supports catching individual timeout exceptions 
def with_timeout(timeout):
    def decorator(decorated):
        @functools.wraps(decorated)
        def inner(*args, **kwargs):
            pool = multiprocessing.pool.ThreadPool(1)
            async_result = pool.apply_async(decorated, args, kwargs)
            try:
                return async_result.get(timeout)
            except multiprocessing.TimeoutError:
                return

        return inner

    return decorator
