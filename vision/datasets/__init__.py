import learn2learn as l2l

def maml_task_wrapper(dataset, n_way, k_shot, k_query, batch_size, length):
    task_generator = l2l.data.task_generator(dataset,)