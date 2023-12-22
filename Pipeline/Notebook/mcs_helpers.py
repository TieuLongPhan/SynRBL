import multiprocessing
from rdkit.Chem import rdRascalMCES

# Worker function
def worker(q, func, args, kwargs):
    result = func(*args, **kwargs)
    q.put(result)

# Function to run another function with timeout using multiprocessing
def run_with_timeout(func, timeout, *args, **kwargs):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(q, func, args, kwargs))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()  # Clean up
        print(f"Function '{func.__name__}' exceeded the timeout of {timeout} seconds.")
        return None
    else:
        return q.get()

def serialize_params(params):
    # Convert RascalOptions settings into a dictionary
    serialized_params = {
        'singleLargestFrag': params.singleLargestFrag,
        'returnEmptyMCES': params.returnEmptyMCES,
        'timeout': params.timeout,
        'similarityThreshold': params.similarityThreshold
    }
    return serialized_params

def deserialize_params(serialized_params):
    # Reconstruct RascalOptions from the dictionary
    params = rdRascalMCES.RascalOptions()
    params.singleLargestFrag = serialized_params['singleLargestFrag']
    params.returnEmptyMCES = serialized_params['returnEmptyMCES']
    params.timeout = serialized_params['timeout']
    params.similarityThreshold = serialized_params['similarityThreshold']
    return params

def find_mces_wrapper(reactant, product, serialized_params):
    params = deserialize_params(serialized_params)
    return rdRascalMCES.FindMCES(reactant, product, params)[0]



# Function to find MCS with a timeout
def find_mces_with_timeout(reactant, product, params, timeout=60):
    serialized_params = serialize_params(params)
    try:
        result = run_with_timeout(find_mces_wrapper, timeout, reactant, product, serialized_params)
        return result if result is not None else None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None