# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Support for FFN inference execution.

Contains implementations of the `BatchExecutor` interface, which takes care
of actually evaluating the FFN predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
try:
  import Queue as queue
except ImportError:  # for Python 3 compat
  import queue
import threading
import time

from concurrent import futures
import numpy as np
import tensorflow as tf
from .inference_utils import timer_counter
from skimage.filters import gaussian
from joblib import Parallel, delayed

# pylint:disable=g-import-not-at-top
try:
  import thread
except ImportError:  # for Python 3 compat
  import _thread as thread
# pylint:enable=g-import-not-at-top
# import lycon


def resize(im, size, interpolation="bilinear"):
    return lycon.resize(im, width=size[1], height=size[0], interpolation=lycon.Interpolation.NEAREST)
    # return imresize(im, size, interpolation=interpolation, mode="F")


class BatchExecutor(object):
  """Base class for FFN executors."""

  def __init__(self, model , counters, batch_size):
    self.model = model
    self.counters = counters

    self.batch_size = batch_size
    self.active_clients = 0

    # Cache input/output sizes.
    self._input_seed_size = np.array(model.input_seed_size[::-1]).tolist()
    self._input_image_size = np.array(model.input_image_size[::-1]).tolist()
    self._pred_size = np.array(model.pred_mask_size[::-1]).tolist()

    self._initialize_model()

  def start_server(self):
    raise NotImplementedError()

  def stop_server(self):
    raise NotImplementedError()

  def start_client(self):
    """Registers a new client.

    Returns:
      client ID
    """
    raise NotImplementedError()

  def finish_client(self, client_id):
    """Deregisters a client."""
    raise NotImplementedError()

  def predict(self, client_id, seed, image, fetches):
    raise NotImplementedError()

  def _initialize_model(self):
    self.model.define_tf_graph()

  def _run_executor(self):
    raise NotImplementedError()

  def _run_executor_log_exceptions(self):
    """Runs the main loop of the executor.

    Logs any exceptions and re-raises them.
    """
    try:
      self._run_executor()
    except Exception as e:  # pylint:disable=broad-except
      logging.exception(e)
      # If the executor fails, the whole process becomes useless and we need
      # to make sure it gets terminated.
      thread.interrupt_main()
      time.sleep(10)
      os._exit(1)  # pylint:disable=protected-access

  @property
  def num_devices(self):
    return 1


class ThreadingBatchExecutor(BatchExecutor):
  """Thread-based BatchExecutor.

  The intended use is to have multiple threads sharing the same executor
  object with:
    - a server thread started with `start_server`
    - each client running in its own thread.

  It is recommended to start the client threads as daemons, so that failures
  of the server thread will result in termination of the whole program.

  Note that the number of clients can (and for efficient utilization of ML
  accelerators, should) exceed the batch size. This makes sense to do even
  if the batch size is 1.
  """

  def __init__(self, model, counters, batch_size, augs=[], expected_clients=1):
    super(ThreadingBatchExecutor, self).__init__(model, counters,
                                                 batch_size)
    self._lock = threading.Lock()
    self.outputs = {}  # Will be populated by Queues as clients register.
    # Used by clients to communiate with the executor. The protocol is
    # as follows:
    #  - 'exit': indicates a request to kill the executor
    #  - N >= 0: registration of a new client with the specified ID
    #  - N < 0: deregistration of an existing client with ID -N - 1
    #  (client_id, seed, image, fetches): request to perform inference
    self.input_queue = queue.Queue()

    # Total clients seen during the lifetime of the executor.
    self.total_clients = 0

    # This many clients need to register themselves during the lifetime of
    # the executor in order for it be allowed to terminate.
    self.expected_clients = expected_clients

    # Arrays fed to TF.
    self.input_seed = np.zeros([batch_size] + self._input_seed_size + [1],
                               dtype=np.float32)
    self.input_image = np.zeros([batch_size] + self._input_image_size + [2],
                                dtype=np.float32)
    self.th_executor = None
    self.augs = augs

  def start_server(self):
    """Starts the server which will evaluate TF models.

    The server will automatically terminate after no more clients are
    registered, and after at least one client has registered and
    deregistered.
    """
    if self.th_executor is None:
      self.th_executor = threading.Thread(
          target=self._run_executor_log_exceptions)
      self.th_executor.start()

  def stop_server(self):
    logging.info('Requesting executor shutdown.')
    self.input_queue.put('exit')
    self.th_executor.join()
    logging.info('Executor shutdown complete.')

  def _run_executor(self):
    """Main loop of the server thread which runs TF code."""
    self._curr_infeed = 0
    logging.info('Executor starting.')
    self.squeeze_shape = np.array(self.input_image.shape[1:])

    with Parallel(n_jobs=32, prefer="threads") as parallel:
      while self.active_clients or self.total_clients < self.expected_clients:
        self.counters['executor-clients'].Set(self.active_clients)

        with timer_counter(self.counters, 'executor-input'):
          ready = []
          while (len(ready) < min(self.active_clients, self.batch_size) or
                 not self.active_clients):
            try:
              data = self.input_queue.get(timeout=5)
            except queue.Empty:
              continue
            if data == 'exit':
              logging.info('Executor shut down requested.')
              return
            elif isinstance(data, int):
              client_id = data
              if client_id >= 0:
                self.total_clients += 1
                self.active_clients += 1
                logging.info('client %d starting', client_id)
              else:
                logging.info('client %d terminating', -client_id - 1)
                self.active_clients -= 1
            else:
              client_id, seed, image, fetches = data
              l = len(ready)
              self.reslicing = None
              # self.input_seed[l, ..., 0] = seed
              # self.input_image[l, ..., :] = image
              self.input_seed[:] = seed[..., None]
              self.input_image = image
              ready.append(client_id)

        if ready:
          self._schedule_batch(ready, fetches, parallel=parallel)

    logging.info('Executor terminating.')

  def _schedule_batch(self, client_ids, fetches, parallel=None):
    """Schedules a single batch for execution."""
    with timer_counter(self.counters, 'executor-inference'):
      try:

        # reshapes = [64, 32, 16]  # 48, 32, 16]  # , 48]  # [64, 56, 48, 32]
        # flips = [(0, 0)]  # , (0, 1), (1, 1)]  # [0, 1]  # [0, 1]
        # add_noise = True

        # from itertools import product
        # from skimage.transform import resize

        seed = self.input_seed
        data = self.input_image

        seed_template = seed[0]
        data_template = data[0]
        data_shape = data_template.shape

        # augs = [x for x in product(reshapes, flips)]
        # augs = []

        if 0:  # len(augs):
          # First augment
          _, _, h, _, _ = data.shape
          aug_data, seeds = [], []
          for aug in augs:
            scale, (flip, flip_dim) = aug
            it_data = np.copy(data_template)
            
            if add_noise:
              noise = np.random.rand(*it_data.shape[:-1]) * 0.1
              noise = np.stack((noise, np.zeros_like(noise)), -1)
              it_data += noise

            if flip:
              it_data = np.flip(it_data, axis=flip_dim)

            if scale != h:
              res_data = parallel(
                delayed(
                  lambda x, y: resize(
                    x,
                    y)  # ,
                    # anti_aliasing=True,
                    # preserve_range=True,
                    # order=1)
                  )(vm, [scale, scale]) for vm in it_data)
              # res_data = []
              # for x in it_data:
              #   res_data.append(
              #     resize(
              #       x,
              #       [scale, scale],
              #       anti_aliasing=True,
              #       preserve_range=True,
              #       order=1)
              #   )
              it_data = np.stack((res_data), 0)
              pad_hw = (data_shape[1] - scale) // 2

              # Now pad
              # print("mins: {} maxs: {}".format(it_data.min((0, 1, 2)), it_data.max((0, 1, 2))))
              # pad_im = np.pad(
              #   it_data[..., [0]],
              #   [[0, 0], [pad_hw, pad_hw], [pad_hw, pad_hw], [0, 0]],
              #   mode="constant",
              #   constant_values=-1.5)
              # pad_mem = np.pad(
              #   it_data[..., [1]],
              #   [[0, 0], [pad_hw, pad_hw], [pad_hw, pad_hw], [0, 0]],
              #   mode="constant",
              #   constant_values=-3.88)
              # -1.5, -3.88, 3.17, 3.84
              it_data = np.pad(
                it_data,
                [[0, 0], [pad_hw, pad_hw], [pad_hw, pad_hw], [0, 0]],
                mode="edge")  # "constant",

              # it_data = np.concatenate((pad_im, pad_mem), -1)
            aug_data.append(it_data)
            seeds.append(seed_template)
          aug_data = np.stack(aug_data, 0)
          seeds = np.stack(seeds, 0)
        else:
          seeds = seed
          aug_data = data

        # ret = self.session.run(fetches, {self.model.input_seed: seed, self.model.input_patches: data})
        ret = self.session.run(fetches, {self.model.input_seed: seeds, self.model.input_patches: aug_data})
        # from matplotlib import pyplot as plt;plt.subplot(223);plt.imshow(ret["logits"][0, 8, ..., 0]);plt.subplot(224);plt.imshow(ret["logits"][1, 8, ..., 0]);plt.subplot(221);plt.imshow(aug_data[0, 8, ..., 0]);plt.subplot(222);plt.imshow(aug_data[1, 8, ..., 0]);plt.show()
        if len(self.augs):
          _, h, _, _ = ret["logits"][0].shape

          # Then unwind augmentations
          logits, new_seeds = [], []
          for it_data, it_seed, aug in zip(ret["logits"], seeds, self.augs):
            scale, (flip, flip_dim) = aug
            # new_seeds.append(it_seed)

            if flip:
              it_data = np.flip(it_data, axis=flip_dim)

            if scale < 1:  # != h:

              # Remove pad
              pad_hw = int((data_shape[1] - (data_shape[1] * scale)) // 2)
              it_data = it_data[:, pad_hw: -pad_hw, pad_hw: -pad_hw]

              # Resize
              # it_data = ndimage.zoom(it_data, [1, data_shape[1], data_shape[2], 1], order=1, prefilter=False)  # , grid_mode=True)
              res_data = parallel(
                delayed(
                  lambda x, y: resize(
                    x,
                    y)  # ,
                    # anti_aliasing=True,
                    # preserve_range=True,
                    # order=1)
                  )(vm, [data_shape[1], data_shape[2]]) for vm in it_data)
              # res_data = []
              # for x in it_data:
              #   res_data.append(
              #     resize(
              #       x,
              #       [data_shape[1], data_shape[2]],
              #       anti_aliasing=True,
              #       preserve_range=True,
              #       order=1)
              #   )
              it_data = np.stack((res_data), 0)[..., None]
            logits.append(it_data)
          logits = np.stack(logits, 0).mean(0, keepdims=True)
          # import pdb;pdb.set_trace()
          # from matplotlib import pyplot as plt;plt.subplot(223);plt.imshow(ret["logits"][0, 8, ..., 0]);plt.subplot(224);plt.imshow(logits[2, ..., 0]);plt.subplot(221);plt.imshow(aug_data[0, 8, ..., 0]);plt.subplot(222);plt.imshow(aug_data[2, 8, ..., 0]);plt.show()

          # seeds = np.stack(new_seeds, 0).mean(0, keepdims=True)
          ret["logits"] = np.ones_like(ret["logits"]) * logits  # replicate our predictions -> batch size

        # logit_shape = ret['logits'].shape
        # # ret['logits'] = (ret['logits'] - ret['logits'].ravel().mean()) / (ret['logits'].std() + 1e-4)
        # ret['logits'] = gaussian(ret['logits'].squeeze().transpose(1, 2, 0), sigma=1.5, multichannel=True, preserve_range=True, truncate=100)  # .transpose(2, 0, 1)
        # ret['logits'] = ret['logits'].transpose(2, 0, 1).reshape(logit_shape)

        if self.reslicing is not None:
          ret['logits'] = ret['logits'][self.reslicing]
        # plt.subplot(131);plt.imshow(self.input_image.squeeze()[5, ..., 0]);plt.colorbar();plt.subplot(132);plt.imshow(self.input_image.squeeze()[5, ..., 1]); plt.colorbar();plt.subplot(133);plt.imshow(ret['logits'].squeeze()[5], vmin=-5, vmax=5);plt.colorbar();plt.show()
      except Exception as e:  # pylint:disable=broad-except
        logging.exception(e)
        # If calling TF didn't work (faulty hardware, misconfiguration, etc),
        # we want to terminate the whole program.
        thread.interrupt_main()
        raise e

    with timer_counter(self.counters, 'executor-output'):
      with self._lock:
        for i, client_id in enumerate(client_ids):
          try:
            self.outputs[client_id].put(
                {k: v[i, ...] for k, v in ret.items()})
          except KeyError:
            # This could happen if a client unregistered itself
            # while inference was running.
            pass

  def start_client(self):
    with self._lock:
      if not self.outputs:
        client_id = 0
      else:
        client_id = max(self.outputs.keys()) + 1

      self.outputs[client_id] = queue.Queue()

    self.input_queue.put(client_id)
    return client_id

  def finish_client(self, client_id):
    self.input_queue.put(-1 - client_id)
    with self._lock:
      del self.outputs[client_id]

  def predict(self, client_id, seed, image, fetches):
    self.input_queue.put((client_id, seed, image, fetches))

    with timer_counter(self.counters, 'client-wait'):
      ret = self.outputs[client_id].get()

    return ret

