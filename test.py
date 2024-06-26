import threading
import time


def sleep_sort(numbers):
    def sleep_and_print(n):
        time.sleep(n)
        print(n, end=' ')

    threads = []
    for number in numbers:
        thread = threading.Thread(target=sleep_and_print, args=(number,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    begin = time.time()
    numbers = [3, 1, 4, 1, 5, 9, 2, 6]
    sleep_sort(numbers)
    end = time.time()
    print(f"Time taken: {end - begin} seconds")