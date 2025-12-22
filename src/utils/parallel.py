"""Parallel execution utility functions extracted from dust3r."""

from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count


def starcall(args):
    """Convenient wrapper for Process.Pool."""
    function, args = args
    return function(*args)


def starstarcall(args):
    """Convenient wrapper for Process.Pool."""
    function, args = args
    return function(**args)


def parallel_threads(function, args, workers=0, star_args=False, kw_args=False, front_num=1, Pool=ThreadPool, **tqdm_kw):
    """tqdm but with parallel execution.

    Will essentially return
      res = [ function(arg) # default
              function(*arg) # if star_args is True
              function(**arg) # if kw_args is True
              for arg in args]

    Args:
        function: Function to apply to each argument.
        args: Iterable of arguments.
        workers: Number of worker threads/processes (0 means use all CPUs).
        star_args: If True, unpack args as *args.
        kw_args: If True, unpack args as **kwargs.
        front_num: Number of first elements to execute sequentially (useful for debugging).
        Pool: Pool class to use (ThreadPool or multiprocessing.Pool).
        **tqdm_kw: Additional keyword arguments for tqdm.

    Returns:
        List of results from applying function to each arg.

    Note:
        The <front_num> first elements of args will not be parallelized.
        This can be useful for debugging.
    """
    while workers <= 0:
        workers += cpu_count()
    if workers == 1:
        front_num = float('inf')

    # convert into an iterable
    try:
        n_args_parallel = len(args) - front_num
    except TypeError:
        n_args_parallel = None
    args = iter(args)

    # sequential execution first
    front = []
    while len(front) < front_num:
        try:
            a = next(args)
        except StopIteration:
            return front  # end of the iterable
        front.append(function(*a) if star_args else function(**a) if kw_args else function(a))

    # then parallel execution
    out = []
    with Pool(workers) as pool:
        # Pass the elements of args into function
        if star_args:
            futures = pool.imap(starcall, [(function, a) for a in args])
        elif kw_args:
            futures = pool.imap(starstarcall, [(function, a) for a in args])
        else:
            futures = pool.imap(function, args)
        # Print out the progress as tasks complete
        for f in tqdm(futures, total=n_args_parallel, **tqdm_kw):
            out.append(f)
    return front + out


def parallel_processes(*args, **kwargs):
    """Same as parallel_threads, with processes.

    Uses multiprocessing.Pool instead of ThreadPool for true parallelism
    across CPU cores rather than cooperative multitasking.
    """
    import multiprocessing as mp
    kwargs['Pool'] = mp.Pool
    return parallel_threads(*args, **kwargs)
