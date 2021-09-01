import numpy as np
import tensorflow as tf

class Server:
  def __init__(self, model_factory, clients_importance_preprocess, weight_delta_aggregator, clients_per_round):
    self._clients_importance_preprocess = clients_importance_preprocess
    self._weight_delta_aggregator = weight_delta_aggregator
    self._clients_per_round = clients_per_round if clients_per_round == 'all' else int(clients_per_round)

    self.model = model_factory()

    self.model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=['accuracy']
    )

  def train(self, clients, test_x, test_y, start_round, num_of_rounds, expr_basename, history, history_delta_sum, progress_callback):
    client2importance = self._clients_importance_preprocess([c.num_of_samples for c in clients])

    server_weights = self.model.get_weights()

    for r in range(start_round, num_of_rounds):
      selected_clients = clients if self._clients_per_round == 'all' \
        else np.random.choice(clients, self._clients_per_round, replace=False)

      deltas = []
      for i, client in enumerate(selected_clients):
        print(f'{expr_basename} round={r + 1}/{num_of_rounds}, client {i + 1}/{self._clients_per_round}',
              end='')

        deltas.append(client.train(server_weights))

        if i != len(selected_clients) - 1:
          print('\r', end='')
        else:
          print('')

      if client2importance is not None:
        importance_weights = [client2importance[c.idx] for c in selected_clients]
      else:
        importance_weights = None
        
      if r==0:
        history_delta_sum = deltas
      else:
        history_delta_sum = [np.add(h, d) for h, d in zip(history_delta_sum, deltas)]
      if 'gamma_mean' in self._weight_delta_aggregator.__name__:
        server_weights = [w + self._weight_delta_aggregator([d[i] for d in deltas], importance_weights, history_points = [np.divide(h[i], r + 1) for h in history_delta_sum])
                        for i, w in enumerate(server_weights)]
      else:
        # todo change code below (to be nicer?):
        # aggregated_deltas = [self._weight_delta_aggregator(_, importance_weights) for _ in zip(*deltas)]
        # server_weights = [w + d for w, d in zip(server_weights, aggregated_deltas)]
        server_weights = [w + self._weight_delta_aggregator([d[i] for d in deltas], importance_weights)
                          for i, w in enumerate(server_weights)]
      

      self.model.set_weights(server_weights)
      loss, acc = self.model.evaluate(test_x, test_y, verbose=0)
      print(f'{expr_basename} loss: {loss} - accuracy: {acc:.2%}')
      history.append((loss, acc))
      if (r + 1) % 10 == 0:
        progress_callback(history, server_weights, history_delta_sum)
