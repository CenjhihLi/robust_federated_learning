import numpy as np
import tensorflow as tf
import random
import gc
class Server:
  def __init__(self, model_factory, weight_delta_aggregator, clients_per_round):
    self._weight_delta_aggregator = weight_delta_aggregator
    self._clients_per_round = clients_per_round if clients_per_round == 'all' else int(clients_per_round)

    self.model = model_factory()

  def train(self, seed, clients, test_x, test_y, start_round, num_of_rounds, expr_basename, history, history_delta_sum,
            optimizer, loss_fn, initial_lr, #last_deltas,
            progress_callback):
    if start_round>1:
      old_loss = history[-1][0]

    self.model.compile(
      loss = loss_fn,
      metrics = ['accuracy']
    )

    loss_descent = True
    server_weights = self.model.get_weights()

    def clip_value(gradient, clip_norm=1):
      if clip_norm == 0:
        clip = False
      else:
        clip = True
      gnorm = np.linalg.norm(np.reshape(gradient,[-1]))
      cfactor = np.minimum(np.divide(clip_norm, gnorm) , 1)  if clip else 1
      return np.multiply(gradient, cfactor) if gnorm > clip_norm else gradient


    for r in range(start_round, num_of_rounds):
      selected_clients = clients if self._clients_per_round == 'all' \
        else np.random.choice(clients, self._clients_per_round, replace=False)
      
      #np.random.seed(seed+r+1)
      #tf.random.set_seed(seed+r+1)
      #random.seed(seed+r+1)

      def decayed_learning_rate(initial_learning_rate, step, decay_steps = 1000, alpha = 0):
        step = min(step, decay_steps-1)
        cosine_decay = 0.5 * (1 + np.cos( np.pi * step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        return initial_learning_rate * decayed
      if loss_descent or r<np.maximum(num_of_rounds*0.5, 500):
        lr_decayed = decayed_learning_rate ( initial_learning_rate = initial_lr, step = r + 1)
      else:
        lr_decayed = 0.9*lr_decayed

      def Adam(lm, lv, d, lr_decayed, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7):
        g = np.multiply( lr_decayed, d) #since the delta return from clients including lr_decayed factor
        m = np.divide(np.multiply( beta_1, lm) + np.multiply( 1-beta_1, g), 1 - np.power(beta_1, r+1) ) 
        v = np.divide(np.multiply( beta_2, lv) + np.multiply( 1-beta_2, np.square(g)), 1 - np.power(beta_2, r+1))  
        d = np.divide( np.multiply( lr_decayed, m),  np.add( np.sqrt(v), epsilon) )
        return m, v, d
      """
      if r>0:
        last_m = last_deltas[0]
        last_v = last_deltas[1]
      else:
        last_m = []
        last_v = []
      """
      deltas = []
      #moments = []
      #velocs = []
      #seems like we need to use the aggregate momentent and velocity 
      for i, client in enumerate(selected_clients):
        print(f'{expr_basename} round={r + 1}/{num_of_rounds}, client {i + 1}/{self._clients_per_round}',
              end='')
        delta = client.train(server_weights, lr_decayed, optimizer, loss_fn)
        #if r > 0:
        #  lms, lvs, delta = zip(*[ Adam(m, v, d, lr_decayed) for m, v, d in zip(last_m[i], last_v[i], delta)])
        #else: 
        #  lms, lvs, delta = zip(*[ Adam(0, 0, d, lr_decayed) for d in delta])
        deltas.append ( delta )
        #moments.append ( lms )
        #velocs.append ( lvs )

        if i != len(selected_clients) - 1:
          print('\r', end='')
        else:
          print('')

      #last_deltas = [moments, velocs]
        
      if r==0:
        history_delta_sum = deltas
      else:
        history_delta_sum = [[np.add(h, d) for h, d in zip(hs, ds)] for hs, ds in zip(history_delta_sum, deltas)]
      
      """
      for i, w in enumerate(server_weights):
        if 'gamma_mean' in self._weight_delta_aggregator.__name__:
          #aggr_delta = clip_value(self._weight_delta_aggregator([d[i] for d in deltas], 
          #              importance_weights, history_points = [np.divide(h[i], r + 1) for h in history_delta_sum]), lr_decayed)
          aggr_delta = self._weight_delta_aggregator([d[i] for d in deltas], 
                        importance_weights, history_points = [np.divide(h[i], r + 1) for h in history_delta_sum])
        else:
          #aggr_delta = clip_value(self._weight_delta_aggregator([d[i] for d in deltas], importance_weights), lr_decayed)
          aggr_delta = self._weight_delta_aggregator([d[i] for d in deltas], importance_weights)
        if r > 0:
          last_m[i], last_v[i], aggr_delta = Adam(last_m[i], last_v[i], aggr_delta, lr_decayed)
        else: 
          lm, lv, aggr_delta = Adam(0, 0, aggr_delta, lr_decayed)
          last_m.append(lm)
          last_v.append(lv)
        #server_weights[i] = w + np.multiply(lr_decayed, aggr_delta)
        server_weights[i] = w + aggr_delta
      last_deltas = [last_m, last_v]
      """
      old_server_weights = server_weights
      if 'record_gamma_mean_' in self._weight_delta_aggregator.__name__:
        server_weights = [w + self._weight_delta_aggregator([d[i] for d in deltas], history_points = [np.divide(h[i], r + 1) for h in history_delta_sum])
                        for i, w in enumerate(server_weights)]
        #server_weights = [w + clip_value(self._weight_delta_aggregator([d[i] for d in deltas], importance_weights, history_points = [np.divide(h[i], r + 1) for h in history_delta_sum]), lr_decayed)
        #                for i, w in enumerate(server_weights)]
        #server_weights = [w + np.multiply(lr_decayed, clip_value(self._weight_delta_aggregator([d[i] for d in deltas], importance_weights, history_points = [np.divide(h[i], r + 1) for h in history_delta_sum])))
        #                for i, w in enumerate(server_weights)]
      else:
        # todo change code below (to be nicer?):
        # aggregated_deltas = [self._weight_delta_aggregator(_, importance_weights) for _ in zip(*deltas)]
        # server_weights = [w + d for w, d in zip(server_weights, aggregated_deltas)]
        server_weights = [w + self._weight_delta_aggregator([d[i] for d in deltas])
                          for i, w in enumerate(server_weights)]
        #server_weights = [w + clip_value(self._weight_delta_aggregator([d[i] for d in deltas], importance_weights), lr_decayed)
        #                  for i, w in enumerate(server_weights)]
        #server_weights = [w + np.multiply(lr_decayed, clip_value(self._weight_delta_aggregator([d[i] for d in deltas], importance_weights)))
        #                  for i, w in enumerate(server_weights)]
      
      self.model.set_weights(server_weights)
      loss, acc = self.model.evaluate(test_x, test_y, verbose=0)
      if r>=np.maximum(num_of_rounds*0.5, 500):
        if loss > old_loss: #need to find some way to avoid going into local minimum
          self.model.set_weights(old_server_weights)
          server_weights = old_server_weights
          loss, acc = self.model.evaluate(test_x, test_y, verbose=0)
          loss_descent=False
        else:
          loss_descent=True
      old_loss = loss
      
      print(f'{expr_basename} loss: {loss} - accuracy: {acc:.2%}')
      history.append((loss, acc))
      if (r + 1) % 10 == 0:
        progress_callback(history, server_weights, history_delta_sum)#, last_deltas)
      gc.collect()
