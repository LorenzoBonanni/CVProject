import multiprocessing

with open('experiments.txt') as f:
    processes = f.readlines()

# Define the maximum number of concurrent processes
max_processes = 1


# Define a function that runs a process using subprocess
def run_process(process):
    """
    Execute a shell command or process.

    This function takes a string representing a shell command or process and executes
    it using the subprocess module. It prints the process to be executed and then
    runs it using the `subprocess.run()` function with `shell=True`.

    :param process: The shell command or process to execute.
    :type process: str
    """
    import subprocess
    print(process)  # Print the process to be executed
    subprocess.run(process, shell=True)  # Execute the process


def main():
    """
    Execute multiple processes concurrently using multiprocessing.

    This function creates a pool of worker processes and executes multiple processes
    concurrently. It submits each process to the pool for execution and waits for
    their completion. Once all processes are completed, it prints a message indicating
    that all processes have finished.
    """
    # Create a pool of max_processes worker processes
    pool = multiprocessing.Pool(max_processes)

    # Create an empty list to store the AsyncResult objects
    results = []

    # Loop through the processes list
    for process in processes:
        # Submit a task to the pool and append the AsyncResult object to the list
        results.append(pool.apply_async(run_process, (process,)))

    # Loop through the results list
    for result in results:
        # Wait for the task to finish and get the result
        result.wait()
        result.get()

    # Close and join the pool
    pool.close()
    pool.join()

    # Print a message when done
    print("All processes completed.")


if __name__ == '__main__':
    main()
