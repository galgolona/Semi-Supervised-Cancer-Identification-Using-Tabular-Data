from unittest import TestCase
import os
import numpy as np
from src.datasets.cancer import CANCERADDataset
import main


class Test(TestCase):
    def test_load_dataset(self):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.split(script_path)[0]
        tr_file = script_dir + '/train_u_with_days.csv'

        dataset_name = 'cancer'
        net_name = 'cancer_mlp'
        xp_path = 'datasets/log/DeepSAD/cancer_test'
        data_path = tr_file
        experiment_name = "25_per_labeled_0_unlabeled"
        iteration_num = 1
        experiment_method = 2
        balanced_train = 1
        balanced_batches = 1
        ratio_known_normal = 1
        ratio_unknown_normal = 0
        ratio_known_outlier = 0.25
        ratio_unknown_outlier = 0.00
        lr = 0.0001
        n_epochs = 70
        lr_milestone = 50
        batch_size = 128
        weight_decay = 0.5e-6
        pretrain = True
        ae_lr = 0.0001
        ae_n_epochs = 70
        ae_batch_size = 128
        ae_weight_decay = 0.5e-3
        normal_class = 0
        known_outlier_class = 1
        n_known_outlier_classes = 1
        index_baseline_name = None
        active_selection_n_samples = 10000
        k_means_chosen_k = 5

        dataset = CANCERADDataset(root=data_path,
                                  dataset_name=dataset_name,
                                  experiment_method=experiment_method,
                                  iteration_num=iteration_num,
                                  experiment_name=experiment_name,
                                  index_baseline_name=index_baseline_name,
                                  active_selection_n_samples=active_selection_n_samples,
                                  k_means_chosen_k=k_means_chosen_k,
                                  balanced_train=balanced_train,
                                  balanced_batches=balanced_batches,
                                  n_known_outlier_classes=n_known_outlier_classes,
                                  ratio_known_normal=ratio_known_normal,
                                  ratio_unknown_normal=ratio_unknown_normal,
                                  ratio_known_outlier=ratio_known_outlier,
                                  ratio_unknown_outlier=ratio_unknown_outlier,
                                  random_state=np.random.RandomState(1000))
        return

    def test_main(self):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.split(script_path)[0]
        tr_file = script_dir + '/train_u_with_days.csv'
        dataset_name = 'cancer'
        net_name = 'cancer_mlp'
        xp_path = 'datasets/log/DeepSAD/cancer_test'
        data_path = tr_file
        experiment_name = "25_per_labeled_0_unlabeled"
        iteration_num = 1
        experiment_method = 2
        balanced_train = 1
        balanced_batches = 1
        ratio_known_normal = 1
        ratio_unknown_normal = 0
        ratio_known_outlier = 0.25
        ratio_unknown_outlier = 0.00
        lr = 0.0001
        n_epochs = 3
        lr_milestone = []
        batch_size = 128
        weight_decay = 0.5e-6
        pretrain = True
        ae_lr = 0.0001
        ae_n_epochs = 2
        ae_batch_size = 128
        ae_weight_decay = 0.5e-3
        normal_class = 0
        known_outlier_class = 1
        n_known_outlier_classes = 1
        index_baseline_name = None
        active_selection_n_samples = 10000
        k_means_chosen_k = 5
        ratio_pollution = 0
        eta = 1.0
        minority_loss = 1.0
        load_model = None
        device = 'cuda'
        seed = -1,
        optimizer_name = 'adam'
        ae_optimizer_name = 'adam'
        ae_lr_milestone = []
        num_threads = 0
        n_jobs_dataloader = 0

        main.main(
            eta=eta,
            ratio_pollution=ratio_pollution,
            data_path=data_path,
            load_config=None,
            net_name=net_name,
            xp_path=xp_path,
            dataset_name=dataset_name,
            experiment_method=experiment_method,
            iteration_num=iteration_num,
            experiment_name=experiment_name,
            index_baseline_name=index_baseline_name,
            active_selection_n_samples=active_selection_n_samples,
            k_means_chosen_k=k_means_chosen_k,
            balanced_train=balanced_train,
            balanced_batches=balanced_batches,
            n_known_outlier_classes=n_known_outlier_classes,
            ratio_known_normal=ratio_known_normal,
            ratio_unknown_normal=ratio_unknown_normal,
            ratio_known_outlier=ratio_known_outlier,
            ratio_unknown_outlier=ratio_unknown_outlier,
            lr=lr,
            n_epochs=n_epochs,
            lr_milestone=lr_milestone,
            batch_size=batch_size,
            weight_decay=weight_decay,
            pretrain=pretrain,
            ae_lr=ae_lr,
            ae_n_epochs=ae_n_epochs,
            ae_batch_size=ae_batch_size,
            ae_weight_decay=ae_weight_decay,
            normal_class=normal_class,
            known_outlier_class=known_outlier_class,
            minority_loss=minority_loss,
            load_model=load_model,
            device=device,
            seed=seed,
            optimizer_name=optimizer_name,
            ae_optimizer_name=ae_optimizer_name,
            ae_lr_milestone=ae_lr_milestone,
            num_threads=num_threads,
            n_jobs_dataloader=n_jobs_dataloader
        )

    def test_drop_method(self):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.split(script_path)[0]
        tr_file = script_dir + '/train_u_with_days.csv'
        dataset_name = 'cancer'
        net_name = 'cancer_mlp'
        xp_path = 'datasets/log/DeepSAD/cancer_test'
        data_path = tr_file
        experiment_name = "25_per_labeled_baseline_5_unlabeled_isolation"
        iteration_num = 1
        experiment_method = 4
        balanced_train = 1
        balanced_batches = 1
        ratio_known_normal = 1
        ratio_unknown_normal = 0
        ratio_known_outlier = 1.00
        ratio_unknown_outlier = 0.00
        lr = 0.0001
        n_epochs = 3
        lr_milestone = []
        batch_size = 128
        weight_decay = 0.5e-6
        pretrain = True
        ae_lr = 0.0001
        ae_n_epochs = 2
        ae_batch_size = 128
        ae_weight_decay = 0.5e-3
        normal_class = 0
        known_outlier_class = 1
        n_known_outlier_classes = 1
        index_baseline_name = "25_per_labeled_0_unlabeled"
        active_selection_n_samples = 13147
        k_means_chosen_k = 5
        ratio_pollution = 0
        eta = 1.0
        minority_loss = 1.0
        load_model = None
        device = 'cuda'
        seed = -1,
        optimizer_name = 'adam'
        ae_optimizer_name = 'adam'
        ae_lr_milestone = []
        num_threads = 0
        n_jobs_dataloader = 0

        main.main(
            eta=eta,
            ratio_pollution=ratio_pollution,
            data_path=data_path,
            load_config=None,
            net_name=net_name,
            xp_path=xp_path,
            dataset_name=dataset_name,
            experiment_name= "25_per_labeled_baseline_5_unlabeled_random",
            experiment_method=experiment_method,
            iteration_num=iteration_num,
            index_baseline_name=index_baseline_name,
            active_selection_n_samples=active_selection_n_samples,
            k_means_chosen_k=k_means_chosen_k,
            balanced_train=balanced_train,
            balanced_batches=balanced_batches,
            n_known_outlier_classes=n_known_outlier_classes,
            ratio_known_normal=ratio_known_normal,
            ratio_unknown_normal=ratio_unknown_normal,
            ratio_known_outlier=ratio_known_outlier,
            ratio_unknown_outlier=ratio_unknown_outlier,
            lr=lr,
            n_epochs=n_epochs,
            lr_milestone=lr_milestone,
            batch_size=batch_size,
            weight_decay=weight_decay,
            pretrain=pretrain,
            ae_lr=ae_lr,
            ae_n_epochs=ae_n_epochs,
            ae_batch_size=ae_batch_size,
            ae_weight_decay=ae_weight_decay,
            normal_class=normal_class,
            known_outlier_class=known_outlier_class,
            minority_loss=minority_loss,
            load_model=load_model,
            device=device,
            seed=seed,
            optimizer_name=optimizer_name,
            ae_optimizer_name=ae_optimizer_name,
            ae_lr_milestone=ae_lr_milestone,
            num_threads=num_threads,
            n_jobs_dataloader=n_jobs_dataloader
        )