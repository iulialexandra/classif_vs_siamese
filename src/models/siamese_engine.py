import os
import logging
import numpy.random as rng
import numpy as np
import time
from signal import SIGINT, SIGTERM
import tensorflow as tf
import tools.utils as util
import tools.visualization as vis
import data_processing.dataset_utils as dat
import keras.backend as K
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from contextlib import redirect_stdout
from keras.optimizers import Adam
from tools.modified_sgd import Modified_SGD
from networks.horizontal_nets import *
from networks.original_nets import *


logger = logging.getLogger("siam_logger")


class SiameseEngine():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.evaluate_every = args.evaluate_every
        self.results_path = args.results_path
        self.model = args.model
        self.num_val_ways = args.n_val_ways
        self.val_trials = args.n_val_tasks
        self.image_dims = args.image_dims
        self.results_path = args.results_path
        self.plot_confusion = args.plot_confusion
        self.plot_training_images = args.plot_training_images
        self.plot_wrong_preds = args.plot_wrong_preds
        self.plot_val_images = args.plot_val_images
        self.plot_test_images = args.plot_test_images
        self.learning_rate = args.learning_rate
        self.lr_annealing = args.lr_annealing
        self.momentum_annealing = args.momentum_annealing
        self.momentum_slope = args.momentum_slope
        self.final_momentum = args.final_momentum
        self.optimizer = args.optimizer
        self.save_weights = args.save_weights
        self.checkpoint = args.chkpt
        self.summary_writer = tf.summary.FileWriter(self.results_path)

    def setup_input(self, sess, class_indices, num_samples, filenames, type):
        new_labels = list(range(len(class_indices)))

        if type == "train":
            dataset_chunk = self.batch_size
            shuffle = True
        else:
            shuffle = False
            if num_samples <= self.val_trials:
                dataset_chunk = num_samples
            else:
                dataset_chunk = min(num_samples, max(2 * len(class_indices), self.val_trials))

        initializer = tf.contrib.lookup.KeyValueTensorInitializer(
            tf.convert_to_tensor(class_indices,
                                 dtype=tf.int32),
            tf.convert_to_tensor(new_labels,
                                 dtype=tf.int32))
        table = tf.contrib.lookup.HashTable(initializer, -1, shared_name=type + "_table")

        (iterator, image_batch, label_batch, labels_one_hot_batch) = dat.deploy_dataset(filenames,
                                                    table,
                                                    dataset_chunk,
                                                    self.image_dims,
                                                    shuffle)
        table.init.run(session=sess)

        if type == "train":
            return iterator, image_batch, label_batch, labels_one_hot_batch
        else:
            sess.run(iterator.initializer)
            ims, labs, labs_one_hot = sess.run([image_batch, label_batch, labels_one_hot_batch])
            return  ims, labs, labs_one_hot


    def setup_network(self, num_classes):
        if self.optimizer == 'sgd':
            optimizer = Modified_SGD(
                lr=self.learning_rate,
                momentum=0.5)
        elif self.optimizer == 'adam':
            optimizer = Adam(self.learning_rate)
        else:
            raise ("optimizer not known")

        model = util.str_to_class(self.model)
        network = model(self.image_dims, optimizer)
        self.net = network.build_net(num_classes)

        with open(os.path.join(self.results_path, 'modelsummary.txt'), 'w') as f:
            with redirect_stdout(f):
                self.net.summary()

        if self.checkpoint:
            self.net.load_weights(self.checkpoint)


    def train(self, train_class_names, val_class_names, test_class_names, train_filenames,
              val_filenames, test_filenames, train_class_indices, val_class_indices,
              test_class_indices, num_val_samples, num_test_samples):


        num_train_cls = len(train_class_indices)
        self.setup_network(num_train_cls)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        train_iterator, train_image_batch,\
        train_label_batch, labels_one_hot_batch = self.setup_input(sess, train_class_indices, None, train_filenames,
                                                                   'train')
        val_inputs, val_targets, val_targets_one_hot= self.setup_input(sess, val_class_indices, num_val_samples,
                                                                       val_filenames, 'val')
        test_inputs, test_targets, test_targets_one_hot= self.setup_input(sess, test_class_indices, num_test_samples,
                                                                          test_filenames, 'test')

        with util.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:
            for epoch in range(self.num_epochs):
                sess.run(train_iterator.initializer)
                batch_index = 0
                train_metrics = defaultdict(list)
                logger.info("Training epoch {} ...".format(epoch))
                while True:
                    try:
                        train_ims, train_labs, train_labs_one_hot = sess.run([train_image_batch, train_label_batch,
                                                                              labels_one_hot_batch])
                        metrics = self.net.train_on_batch(x={"input": train_ims},
                                                          y={"classification":
                                                                 train_labs_one_hot})

                        for idx, metric in enumerate(metrics):
                            train_metrics[self.net.metrics_names[idx]].append(metric)

                        batch_index += 1

                    except tf.errors.OutOfRangeError:
                        if self.lr_annealing:
                            K.set_value(self.net.optimizer.lr, K.get_value(
                                self.net.optimizer.lr) * 0.99)

                        if self.momentum_annealing and self.optimizer == 'sgd':
                            if K.get_value(self.net.optimizer.momentum) < self.final_momentum:
                                K.set_value(self.net.optimizer.momentum, K.get_value(
                                    self.net.optimizer.momentum) + self.momentum_slope)

                        if epoch % self.evaluate_every == 0:
                            self.validate(epoch, batch_index, train_metrics, val_inputs, val_targets,
                                          val_targets_one_hot, val_class_names, test_inputs,
                                          test_targets, test_targets_one_hot, test_class_names)

                        break
                if u.interrupted:
                    logger.info("Interrupted on request, doing one last evaluation")
                    self.validate(epoch, batch_index, train_metrics, val_inputs, val_targets, val_targets_one_hot,
                                  val_class_names, test_inputs, test_targets, test_targets_one_hot,
                                  test_class_names)
                    break

    def validate(self, epoch, batch_index, train_metrics, val_inputs, val_targets, val_targets_one_hot, val_class_names,
                 test_inputs, test_targets, test_targets_one_hot, test_class_names):
        train_loss = np.mean(train_metrics["loss"])
        train_acc = np.mean(train_metrics["acc"])

        logger.info("Train classification loss and accuracy at the end of"
                    " epoch {}: {}, {}".format(epoch, train_loss, train_acc))

        epoch_folder = os.path.join(self.results_path, "epoch_{}".format(epoch))
        if not os.path.exists(epoch_folder):
            os.makedirs(epoch_folder)

        val_accuracy, val_y_pred, val_predictions, val_probs_std, val_probs_means, mean_delay, std_delay = self.eval(
            val_inputs,
            val_targets,
            val_class_names)
        test_accuracy, test_y_pred, test_predictions, test_probs_std, test_probs_means, mean_delay, std_delay = self.eval(
            test_inputs,
            test_targets,
            test_class_names)

        util.metrics_to_csv(os.path.join(epoch_folder, "metrics_epoch_{}.csv"
                                                       "".format(epoch)),
                            np.asarray([train_loss, train_acc, val_accuracy,
                                        test_accuracy, mean_delay, std_delay]),
                            ["train_loss", "train_acc", "val_accuracy", "test_accuracy", "mean_delay", "std_delay"]
                            )

        if self.save_weights:
            # self.net.save_weights(os.path.join(self.results_path, "weights.h5"))
            self.net.save(os.path.join(self.results_path, "weights.h5"), overwrite=True, include_optimizer=False)


    def test(self, test_class_names, test_filenames, train_class_indices, test_class_indices, num_test_samples):
        num_train_cls = len(train_class_indices)
        self.setup_network(num_train_cls)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        test_inputs, test_targets, test_targets_one_hot= self.setup_input(sess, test_class_indices, num_test_samples,
                                                                          test_filenames, 'test')

        test_accuracy, test_y_pred, test_predictions, \
        test_probs_std, test_probs_means, delay, std_delay = self.eval(test_inputs, test_targets, test_class_names)

        np.savetxt(os.path.join(self.results_path, "inference.csv"), np.asarray([test_accuracy, delay, std_delay]),
                   ["test_acc", "mean_delay", "std_delay"])

    def eval(self, inps, targets, class_names):

        num_trials = len(inps)
        logger.info(
            "Evaluating model on {} trials from classes {}"
            "...".format(num_trials, class_names))

        y_pred = np.zeros((num_trials))
        probs_std = np.zeros((num_trials))
        probs_means = np.zeros((num_trials))
        timings = np.zeros((num_trials))

        for trial in range(num_trials ):
            start = time.time()
            probs = self.net.predict(np.expand_dims(inps[trial], 0))
            timings[trial] = time.time() - start
            y_pred[trial] = np.argmax(probs)
            probs_std[trial] = np.std(probs)
            probs_means[trial] = np.mean(probs)

        interm_acc = np.equal(y_pred, targets)
        tolerance = probs_std > 10e-8
        preds = np.logical_and(interm_acc, tolerance)
        accuracy = np.mean(preds)
        mean_delay = np.mean(timings[1:])
        std_delay = np.std(timings[1:])

        logger.info("accuracy: {}% on classes {}"
                    "".format(accuracy*100, class_names))
        return accuracy, y_pred, preds, probs_std, probs_means, mean_delay, std_delay

