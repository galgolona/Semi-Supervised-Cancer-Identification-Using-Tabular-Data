import click
import torch
import logging
import random
import numpy as np
import cvxopt as co

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from baselines.ssad import SSAD
from datasets.main import load_dataset


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'fmnist', 'cifar10', 'arrhythmia', 'cardio', 'satellite',
                                                   'satimage-2', 'shuttle', 'thyroid']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--ratio_known_normal', type=float, default=0.0,
              help='Ratio of known (labeled) normal training examples.')
@click.option('--ratio_unknown_normal', type=float, default=0.0,
              help='Ratio of unknown (unlabeled) normal training examples.')
@click.option('--ratio_known_outlier', type=float, default=0.0,
              help='Ratio of known (labeled) anomalous training examples.')
@click.option('--ratio_unknown_outlier', type=float, default=0.0,
              help='Ratio of unknown (unlabeled) anomalous training examples.')
@click.option('--ratio_pollution', type=float, default=0.0,
              help='Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--kernel', type=click.Choice(['rbf']), default='rbf', help='Kernel for SSAD')
@click.option('--kappa', type=float, default=1.0, help='SSAD hyperparameter kappa.')
@click.option('--hybrid', type=bool, default=False,
              help='Train SSAD on features extracted from an autoencoder. If True, load_ae must be specified')
@click.option('--load_ae', type=click.Path(exists=True), default=None,
              help='Model file path to load autoencoder weights (default: None).')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--known_outlier_class', type=int, default=1,
              help='Specify the known outlier class of the dataset for semi-supervised anomaly detection.')
@click.option('--n_known_outlier_classes', type=int, default=1,
              help='Number of known outlier classes.'
                   'If 0, no anomalies are known.'
                   'If 1, outlier class as specified in --known_outlier_class option.'
                   'If > 1, the specified number of outlier classes will be sampled at random.')
@click.option('--experiment_name', type=int, default=1,
              help='Specify the name of the file to save selected data semi-supervised setting to reload later that setting')
@click.option('--experiment_method', type=int, default=1,
              help='Specify the method you want to construct the semi supervised setting')
@click.option('--balanced_train', type=int, default=1,
              help='Specify if you would like to use smote before training to enrich train set. '
                   '1 for applying smote 0 otherwise')
@click.option('--balanced_batches', type=int, default=1,
              help='Specify if you would like to use balanced batches in training. '
                   '1 for applying balanced batches 0 otherwise')
@click.option('--k_means_chosen_k', type=int, default=1,
              help='when applying k means in train reconstruction process specify the k. ')
@click.option('--active_selection_n_samples', type=int, default=1,
              help='when applying selecting specific number of samples in train reconstruction process either randomly or kmeans specify the size.')

def main(dataset_name, xp_path, data_path, load_config, load_model,
         experiment_name, iteration_num, experiment_method, balanced_train,
         balanced_batches, ratio_known_normal, ratio_unknown_normal,
         ratio_known_outlier, ratio_unknown_outlier, active_selection_n_samples, k_means_chosen_k,
         ratio_pollution, seed, kernel, kappa, hybrid, load_ae, n_jobs_dataloader,
         normal_class, known_outlier_class, n_known_outlier_classes):

    """
    (Hybrid) SSAD for anomaly detection as in Goernitz et al., Towards Supervised Anomaly Detection, JAIR, 2013.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Ratio of labeled normal train samples: %.2f' % ratio_known_normal)
    logger.info('Ratio of labeled anomalous samples: %.2f' % ratio_known_outlier)
    logger.info('Pollution ratio of unlabeled train data: %.2f' % ratio_pollution)
    if n_known_outlier_classes == 1:
        logger.info('Known anomaly class: %d' % known_outlier_class)
    else:
        logger.info('Number of known anomaly classes: %d' % n_known_outlier_classes)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print SSAD configuration
    logger.info('SSAD kernel: %s' % cfg.settings['kernel'])
    logger.info('Kappa-paramerter: %.2f' % cfg.settings['kappa'])
    logger.info('Hybrid model: %s' % cfg.settings['hybrid'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        co.setseed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Use 'cpu' as device for SSAD
    device = 'cpu'
    torch.multiprocessing.set_sharing_strategy('file_system')  # fix multiprocessing issue for ubuntu
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class,
                           experiment_name, iteration_num, experiment_method, balanced_train,
                           balanced_batches, n_known_outlier_classes,
                           ratio_known_normal, ratio_unknown_normal,
                           ratio_known_outlier, ratio_unknown_outlier, active_selection_n_samples,
                           k_means_chosen_k, ratio_pollution,
                           random_state=np.random.RandomState(cfg.settings['seed']))

    # Log random sample of known anomaly classes if more than 1 class
    if n_known_outlier_classes > 1:
        logger.info('Known anomaly classes: %s' % (dataset.known_outlier_classes,))

    # Initialize SSAD model
    ssad = SSAD(kernel=cfg.settings['kernel'], kappa=cfg.settings['kappa'], hybrid=cfg.settings['hybrid'])

    # If specified, load model parameters from already trained model
    if load_model:
        ssad.load_model(import_path=load_model, device=device)
        logger.info('Loading model from %s.' % load_model)

    # If specified, load model autoencoder weights for a hybrid approach
    if hybrid and load_ae is not None:
        ssad.load_ae(dataset_name, model_path=load_ae)
        logger.info('Loaded pretrained autoencoder for features from %s.' % load_ae)

    # Train model on dataset
    ssad.train(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    ssad.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)
    #convert method num to text
    if experiment_method==0:
        method = "active_selection_n_labeled_farthest_point"
    if experiment_method==1:
        method = "active_selection_n_labeled_random"
    if experiment_method==2:
        method = "active_selection_n_labeled_kmeans" + k_means_chosen_k
    if experiment_method == 3:
        method = "semi_proportions"
    if experiment_method == 4:
         method = "unlabeled_add_isolation_forest"
    if experiment_method == 5:
         method = "unlabeled_add_random"
    if experiment_method == 6:
          method = "semi_proportions"

            # Save results and configuration
    ssad.save_results(export_json=xp_path + experiment_name + method + 'bal_batches'+ balanced_batches +
                                  + 'bal_train' + balanced_train + '/results.json')
    cfg.save_config(export_json=xp_path + experiment_name + '/config.json')

    # Plot most anomalous and most normal test samples
    indices, labels, scores = zip(*ssad.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    idx_all_sorted = indices[np.argsort(scores)]  # from lowest to highest score
    idx_normal_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # from lowest to highest score

    if dataset_name in ('mnist', 'fmnist', 'cifar10'):

        if dataset_name in ('mnist', 'fmnist'):
            X_all_low = dataset.test_set.data[idx_all_sorted[:32], ...].unsqueeze(1)
            X_all_high = dataset.test_set.data[idx_all_sorted[-32:], ...].unsqueeze(1)
            X_normal_low = dataset.test_set.data[idx_normal_sorted[:32], ...].unsqueeze(1)
            X_normal_high = dataset.test_set.data[idx_normal_sorted[-32:], ...].unsqueeze(1)

        if dataset_name == 'cifar10':
            X_all_low = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[:32], ...], (0, 3, 1, 2)))
            X_all_high = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[-32:], ...], (0, 3, 1, 2)))
            X_normal_low = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[:32], ...], (0, 3, 1, 2)))
            X_normal_high = torch.tensor(
                np.transpose(dataset.test_set.data[idx_normal_sorted[-32:], ...], (0, 3, 1, 2)))

        plot_images_grid(X_all_low, export_img=xp_path + '/all_low', padding=2)
        plot_images_grid(X_all_high, export_img=xp_path + '/all_high', padding=2)
        plot_images_grid(X_normal_low, export_img=xp_path + '/normals_low', padding=2)
        plot_images_grid(X_normal_high, export_img=xp_path + '/normals_high', padding=2)


if __name__ == '__main__':
    main()
