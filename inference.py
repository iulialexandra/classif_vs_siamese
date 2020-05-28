import argparse
import os
import json
import time
import numpy.random as rng
import tools.utils as util
import data_processing.dataset_utils as dat
import numpy as np
from SiameseEngine import SiameseEngine
import tensorflow as tf
import sys

def _make_oneshot_task(n_val_tasks, image_data, labels, n_ways):
    with tf.device('/cpu:0'):
        classes = np.unique(labels)
        assert len(classes) == n_ways
        if len(image_data) < n_val_tasks:
            replace = True
        else:
            replace = False
        reference_indices = rng.choice(range(len(labels)), size=(n_val_tasks,), replace=replace)
        reference_labels = np.ravel(labels[reference_indices])
        comparison_indices = np.zeros((n_val_tasks, n_ways), dtype=np.int32)
        targets = np.zeros((n_val_tasks, n_ways))
        targets[range(n_val_tasks), reference_labels] = 1
        for i, cls in enumerate(classes):
            cls_indices = np.where(labels == cls)[0]
            comparison_indices[:, i] = rng.choice(cls_indices, size=(n_val_tasks,),
                                                  replace=True)
        comparison_images = image_data[comparison_indices, :, :, :]
        reference_images = image_data[reference_indices, np.newaxis, :, :, :]
        reference_images = np.repeat(reference_images, n_ways, axis=1)
        image_pairs = [np.array(reference_images, dtype=np.float32),
                       np.array(comparison_images, dtype=np.float32)]
        return image_pairs, targets


def inference(args):
    (train_class_names, val_class_names, test_class_names, train_filenames,
    val_filenames, test_filenames, train_class_indices, val_class_indices,
    test_class_indices, num_val_samples, num_test_samples) = dat.read_dataset_csv(args.dataset_path, args.n_val_ways)

    num_train_cls = len(train_class_indices)
    new_val_labels = list(range(len(val_class_indices)))
    if num_val_samples <= args.n_val_tasks:
        val_dataset_chunk = num_val_samples
    else:
        val_dataset_chunk = min(num_val_samples, max(2 * len(val_class_indices), args.n_val_tasks))

    val_init = tf.contrib.lookup.KeyValueTensorInitializer(
        tf.convert_to_tensor(val_class_indices,
                             dtype=tf.int32),
        tf.convert_to_tensor(new_val_labels,
                             dtype=tf.int32))
    val_table = tf.contrib.lookup.HashTable(val_init, -1, shared_name="val_table")
    (val_iterator, val_image_batch,
     val_label_batch, _) = dat.deploy_dataset(val_filenames,
                                              val_table,
                                              val_dataset_chunk,
                                              args.image_dims,
                                              False)

    model = util.str_to_class(args.model)
    siamese_model = model(args.image_dims, args.optimizer,
                          args.left_classif_factor,
                          args.right_classif_factor)
    net = siamese_model.build_net(num_train_cls)

    net.load_weights(os.path.join(args.checkpoint, "weights.h5"))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(val_iterator.initializer)

    val_table.init.run(session=sess)

    val_ims, val_labs = sess.run([val_image_batch, val_label_batch])
    val_inputs, val_targets_one_hot = _make_oneshot_task(args.n_val_tasks, val_ims, val_labs, args.n_val_ways)
    val_targets = np.argmax(val_targets_one_hot, axis=1)

    def eval_func(inps, targets, class_names):

        logger.info(
            "Evaluating model on {} random {} way one-shot learning tasks from classes {}"
            "...".format(args.n_val_tasks, args.n_val_ways, class_names))

        y_pred = np.zeros((args.n_val_tasks))
        probs_std = np.zeros((args.n_val_tasks))
        probs_means = np.zeros((args.n_val_tasks))
        timings = np.zeros((args.n_val_tasks))
        for trial in range(args.n_val_tasks):
            start = time.time()
            probs = net.predict([inps[0][trial], inps[1][trial]])[1]
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
        logger.info("{} way one-shot accuracy: {}% on classes {}, calculated in {} +- {} seconds "
                    "".format(args.n_val_ways, accuracy * 100, class_names, mean_delay, std_delay))
        return accuracy, y_pred, preds, probs_std, probs_means, mean_delay, std_delay

    val_accuracy, val_y_pred, val_predictions, val_probs_std, val_probs_means, delay, std_delay = eval_func(
        val_inputs,
        val_targets,
        val_class_names)

    np.savetxt(os.path.join(args.results_path, "inference.csv"), np.asarray([val_accuracy, delay, std_delay]))
    sys.exit()


def parse_args():
    """Parses arguments specified on the command-line
    """
    argparser = argparse.ArgumentParser('Train and evaluate  siamese networks')

    argparser.add_argument('--batch_size', type=int,
                           help="The number of images to process at the same time",
                           default=32)
    argparser.add_argument('--n_val_tasks', type=int,
                           help="how many one-shot tasks to validate on",
                           default=1000)
    argparser.add_argument('--n_val_ways', type=int,
                           help="how many support images we have for each image to be classified",
                           default=5)
    argparser.add_argument('--num_epochs', type=int,
                           help="Number of training epochs",
                           default=200)
    argparser.add_argument('--evaluate_every', type=int,
                           help="interval for evaluating on one-shot tasks",
                           default=5)
    argparser.add_argument('--momentum_slope', type=float,
                           help="linear epoch slope evolution",
                           default=0.01)
    argparser.add_argument('--final_momentum', type=float,
                           help="Final layer-wise momentum (mu_j in the paper)",
                           default=0.9)
    argparser.add_argument('--learning_rate', type=float,
                           default=0.001)
    argparser.add_argument('--seed', help="Random seed to make experiments reproducible",
                           type=int, default=13)
    argparser.add_argument('--left_classif_factor', help="How much left classification loss is"
                                                         " weighted in the total loss",
                           type=float, default=0.7)
    argparser.add_argument('--right_classif_factor', help="How much right classification loss is"
                                                          " weighted in the total loss",
                           type=float, default=0.7)
    argparser.add_argument('--lr_annealing',
                           help="If set to true, it changes the learning rate at each epoch",
                           type=bool, default=True)
    argparser.add_argument('--momentum_annealing',
                           help="If set to true, it changes the momentum at each epoch",
                           type=bool, default=True)
    argparser.add_argument('--optimizer',
                           help="The optimizer to use for training",
                           type=str, default='sgd')
    argparser.add_argument('--console_print',
                           help="If set to true, it prints logger info to console.",
                           type=bool, default=False)
    argparser.add_argument('--plot_training_images',
                           help="If set to true, it plots input training data",
                           type=bool, default=False)
    argparser.add_argument('--plot_val_images',
                           help="If set to true, it plots input validation data",
                           type=bool, default=False)
    argparser.add_argument('--plot_test_images',
                           help="If set to true, it plots input test data",
                           type=bool, default=False)
    argparser.add_argument('--plot_confusion',
                           help="If set to true, it plots the confusion matrix",
                           type=bool, default=False)
    argparser.add_argument('--plot_wrong_preds',
                           help="If set to true, it plots the images that were predicted wrongly",
                           type=bool, default=False)
    argparser.add_argument('--results_path',
                           help="Path to results. If none, the folder gets created with"
                                "current date and time", default=None)
    argparser.add_argument('--checkpoint',
                           help="Path where the weights to load are",
                           default=None)
    argparser.add_argument('--model', type=str, default="OriginalNetworkV1")
    argparser.add_argument('--save_weights',
                           help="Whether to save the weights at every evaluation",
                           type=bool, default=True)
    argparser.add_argument('--write_to_tensorboard',
                           help="Whether to save the results in a tensorboard-readable format",
                           type=bool, default=False)
    argparser.add_argument('--dataset',
                           help="The dataset of choice", type=str,
                           default="omniglot")
    argparser.add_argument('--data_path',
                           help="Path to data", type=str,
                           default="/mnt/data/datasets/siamese_cluster_data_new")
    return argparser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plotting = False
    args.plot_confusion = plotting
    args.plot_training_images = plotting
    args.plot_val_images = plotting
    args.plot_test_images = plotting
    args.plot_wrong_preds = plotting

    args.write_to_tensorboard = False
    args.save_weights = True
    args.console_print = True
    args.num_epochs = 200
    args.n_val_ways = 5
    args.evaluate_every = 10
    args.n_val_tasks = 1000
    args.batch_size = 16

    args.final_momentum = 0.9
    args.momentum_slope = 0.01
    args.learning_rate = 0.001
    args.lr_annealing = True
    args.momentum_annealing = True
    args.optimizer = "sgd"

    args.left_classif_factor = 0.7
    args.right_classif_factor = 0.7
    # args.dataset = "omniglot"
    # # args.image_dims = (64, 64, 1)
    # args.model = "OriginalNetworkV3"
    # args.data_path = "/mnt/data/siamese_cluster_new/data"
    # args.checkpoint = "./results/2020_5_27-15_31_35_939937_seed_13_omniglot_OriginalNetworkV3_yes"

    if args.dataset == "mnist":
        args.image_dims = (28, 28, 1)
    elif args.dataset == "omniglot":
        args.image_dims = (105, 105, 1)
    elif args.dataset == "cifar100":
        args.image_dims = (32, 32, 3)
    elif args.dataset == "roshambo":
        args.image_dims = (64, 64, 1)
    elif args.dataset == "tiny-imagenet":
        args.image_dims = (64, 64, 3)
    elif args.dataset == "mini-imagenet":
        args.image_dims = (84, 84, 3)
    else:
        print(" Dataset not supported.")

    args.dataset_path = os.path.join(args.data_path, args.dataset)
    args, logger = util.initialize_experiment(args, train=False)
    inference(args)
