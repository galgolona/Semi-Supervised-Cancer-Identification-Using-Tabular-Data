import click
import torch
import logging
import random
import numpy as np
import os
from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from DeepSAD import DeepSAD
from datasets.main import load_dataset

################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'fmnist', 'cifar10', 'arrhythmia', 'cardio', 'satellite',
                                                    'satimage-2', 'shuttle', 'thyroid','cancer','breast_c_kaggle']))
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'fmnist_LeNet', 'cifar10_LeNet', 'arrhythmia_mlp',
                                                'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
                                                'thyroid_mlp','cancer_mlp']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path())
@click.option('--load_config', type=click.Path(exists=True), default=None,
               help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
               help='Model file path (default: None).')
@click.option('--eta', type=float, default=1.0, help='Deep SAD hyperparameter eta (must be 0 < eta).')
@click.option('--ratio_known_normal', type=float, default=0.0,
               help='Ratio of known (labeled) normal training examples.')
@click.option('--ratio_known_outlier', type=float, default=0.0,
               help='Ratio of known (labeled) anomalous training examples.')
@click.option('--ratio_unknown_normal', type=float, default=0.0,
              help='Pollution ratio of unlabeled training data with unknown (unlabeled) normal.'
              'If 0, all normal data in the trainingmodel is known. the ratio of normal is defined by ratio known normal variable.')
@click.option('--ratio_unknown_outlier', type=float, default=0.0,
              help='ratio of unlabeled in outlier training data with unknown (unlabeled) outlier.'
              'If 0, all normal data in the trainingmodel is known. the ratio of normal is defined by ratio known normal variable.')
@click.option('--ratio_pollution', type=float, default=0.0,
               help='Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
               help='Name of the optimizer to use for Deep SAD network training.')
@click.option('--lr', type=float, default=0.001,
               help='Initial learning rate for Deep SAD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
               help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
               help='Weight decay (L2 penalty) hyperparameter for Deep SAD objective.')
@click.option('--pretrain', type=bool, default=True,
               help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam']), default='adam',
               help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
               help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=0, multiple=True,
               help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
               help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--num_threads', type=int, default=0,
               help='Number of threads used for parallelizing CPU operations. 0 means that all resources are used.')
@click.option('--n_jobs_dataloader', type=int, default=0,
               help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
               help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--known_outlier_class', type=int, default=1,
               help='Specify the known outlier class of the dataset for semi-supervised anomaly detection.')
@click.option('--n_known_outlier_classes', type=int, default=0,
               help='Number of known outlier classes.'
                    'If 0, no anomalies are known.'
                    'If 1, outlier class as specified in --known_outlier_class option.'
                    'If > 1, the specified number of outlier classes will be sampled at random.')
@click.option('--balanced_train', type=int, default=1,
               help='Determine whether the train will have equal normal and anomalies sizes.'
                   'If 0, train will be original sample.'
                   'If 1, train will selected balanced.')
@click.option('--balanced_batches', type=int, default=1,
               help='Determine whether in the training process each batch will have equal size of normal and anomalies.'
                   'If 0, training batches will be based on sample.'
                   'If 1, training batches will be balanced.')
@click.option('--minority_loss', type=int, default=1,
               help='Determine whether in the training process the loss function will include enhancement of minority prevalence.'
                   'If 0, deepsad loss funcation is applied.'
                   'If 1, minority loss function is applied (default).')
@click.option('--experiment_name', type=str, default="",
               help='set name for semi supervised split, labeled and size ratio, that will be saved for recurring experiments.')
@click.option('--index_baseline_name', type=str, default="",
               help='define the basline name for semi supervised split, labeled and size ratio, indices will be extracted based on iteration number and name.')
@click.option('--experiment_method', type=int, default=1,
               help='using the semi supervised data split defined in experiment name selected the method to be applied on top of it.')
@click.option('--iteration_num', type=int, default=1,
               help='run number of experiment name and experiment method. based on recurring experiments.')
@click.option('--k_means_chosen_k', type=int, default=5,
               help='in case experiment methood is k means, defines the k.')
@click.option('--active_selection_n_samples', type=int, default=1000,
             help='number of samples to be selected to attend in the experiment.')

def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, eta,
         experiment_name, index_baseline_name, iteration_num, experiment_method, balanced_train, balanced_batches, minority_loss,
         ratio_known_normal, ratio_unknown_normal, ratio_known_outlier, ratio_unknown_outlier, ratio_pollution,
         active_selection_n_samples, k_means_chosen_k, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay,
         pretrain, ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay,
         num_threads, n_jobs_dataloader, normal_class, known_outlier_class, n_known_outlier_classes):

    """
    Deep SAD, a method for deep semi-supervised anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """
    print('method', experiment_method)
    # Get configuration
    cfg = Config(locals().copy())
    script_path = os.path.abspath(__file__)
    script_dir = os.path.split(script_path)[0]
    os.chdir(script_dir)
    xp_path = script_dir + "/log/DeepSAD/cancer_test"

    #os.chdir("/content/drive/My Drive/cancer full files/master_thesis")
    #xp_path = "/content/drive/My Drive/cancer full files/master_thesis/src/log/DeepSAD/cancer_test"
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
    logger.info('Load config is %s' % load_config)
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % data_path)
    logger.info('Export path is %s' % xp_path)

    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Known outlier selected by farthest point algo: %d' % False)
    logger.info('Ratio of labeled normal train samples: %.2f' % ratio_known_normal)
    logger.info('Ratio of unlabeled normal train samples: %.2f' % ratio_unknown_normal)
    logger.info('Ratio of labeled anomalous samples: %.2f' % ratio_known_outlier)
    logger.info('Pollution ratio of unlabeled train data: %.2f' % ratio_pollution)
    if n_known_outlier_classes == 1:
        logger.info('Known anomaly class: %d' % known_outlier_class)
    else:
        logger.info('Number of known anomaly classes: %d' % n_known_outlier_classes)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print model configuration
    logger.info('Eta-parameter: %.2f' % cfg.settings['eta'])

    # Set seed
    if cfg.settings['seed'] != -1 and cfg.settings['seed'] != tuple([-1]):
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    # Set the number of threads used for parallelizing CPU operations
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    logger.info('Computation device: %s' % device)
    logger.info('Number of threads: %d' % num_threads)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(net_name, dataset_name, data_path, normal_class, known_outlier_class,
                 experiment_name, index_baseline_name, iteration_num, experiment_method, balanced_train,
                 balanced_batches, n_known_outlier_classes, ratio_known_normal,
                 ratio_unknown_normal, ratio_known_outlier, ratio_unknown_outlier,
                 active_selection_n_samples, k_means_chosen_k, ratio_pollution,
                 random_state=np.random.RandomState(1000)) #TODO change backcfg.settings['seed']

    # Log random sample of known anomaly classes if more than 1 class
    if n_known_outlier_classes > 1:
        logger.info('Known anomaly classes: %s' % (dataset.known_outlier_classes,))

    # Initialize DeepSAD model and set neural network phi
    deepSAD = DeepSAD(cfg.settings['eta'])
    deepSAD.set_network(net_name)

    # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
    if load_model:
        deepSAD.load_model(model_path=load_model, load_ae=True, map_location=device)
        logger.info('Loading model from %s.' % load_model)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # Pretrain model on dataset (via autoencoder)
        deepSAD.pretrain(dataset,
                         optimizer_name=cfg.settings['ae_optimizer_name'],
                         lr=cfg.settings['ae_lr'],
                         n_epochs=cfg.settings['ae_n_epochs'],
                         lr_milestones=cfg.settings['ae_lr_milestone'],
                         batch_size=cfg.settings['ae_batch_size'],
                         weight_decay=cfg.settings['ae_weight_decay'],
                         device=device,
                         n_jobs_dataloader=n_jobs_dataloader)

        # Save pretraining results
        deepSAD.save_ae_results(export_json=xp_path + '/ae_results.json')

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    deepSAD.train(dataset,
                  optimizer_name=cfg.settings['optimizer_name'],
                  lr=cfg.settings['lr'],
                  n_epochs=cfg.settings['n_epochs'],
                  lr_milestones=cfg.settings['lr_milestone'],
                  batch_size=cfg.settings['batch_size'],
                  weight_decay=cfg.settings['weight_decay'],
                  device=device,
                  n_jobs_dataloader=n_jobs_dataloader,
                  minority_loss=minority_loss)

    # Test model
    deepSAD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)
    # Save results, model, and configuration
    deepSAD.save_results(export_json=xp_path +'/'+ experiment_name + '_method_' + str(experiment_method) + '/results.json')
    deepSAD.save_model(export_model=xp_path + '/model.tar')
    cfg.save_config(export_json=xp_path + '/config.json')

    # Plot most anomalous and most normal test samples
    #indices, labels, scores = zip(*deepSAD.results['test_scores'])
    #indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    #idx_all_sorted = indices[np.argsort(scores)]  # from lowest to highest score
    #idx_normal_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # from lowest to highest score

    if dataset_name in ('mnist', 'fmnist', 'cifar10'):

        if dataset_name in ('mnist', 'fmnist'):
            X_all_low = dataset.test_set.data[idx_all_sorted[:32], ...].unsqueeze(1)
            X_all_high = dataset.test_set.data[idx_all_sorted[-32:], ...].unsqueeze(1)
            X_normal_low = dataset.test_set.data[idx_normal_sorted[:32], ...].unsqueeze(1)
            X_normal_high = dataset.test_set.data[idx_normal_sorted[-32:], ...].unsqueeze(1)

        if dataset_name == 'cifar10':
            X_all_low = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[:32], ...], (0,3,1,2)))
            X_all_high = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[-32:], ...], (0,3,1,2)))
            X_normal_low = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[:32], ...], (0,3,1,2)))
            X_normal_high = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[-32:], ...], (0,3,1,2)))

        plot_images_grid(X_all_low, export_img=xp_path + '/all_low', padding=2)
        plot_images_grid(X_all_high, export_img=xp_path + '/all_high', padding=2)
        plot_images_grid(X_normal_low, export_img=xp_path + '/normals_low', padding=2)
        plot_images_grid(X_normal_high, export_img=xp_path + '/normals_high', padding=2)


if __name__ == '__main__':
    main()
