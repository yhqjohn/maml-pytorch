import learn2learn as l2l


def task_loader(dataset, n_way, k_shot, k_query, length, batch_size=None):
    if batch_size is None:
        generator = l2l.data.TaskGenerator(dataset, n_way, k_shot + k_query, tasks=length)
        for _ in range(length):
            support_set = generator.sample(k_shot)
            query_set = generator.sample(k_query, support_set.sampled_task)
            yield support_set, query_set
        return
    else:
        generator = l2l.data.TaskGenerator(dataset, n_way, k_shot + k_query, tasks=length*batch_size)
        for _ in range(length):
            batch = []
            for __ in range(batch_size):
                support_set = generator.sample(k_shot)
                query_set = generator.sample(k_query, support_set.sampled_task)
                batch.append((support_set, query_set))
            yield batch
        return
