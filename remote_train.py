from clearml import Task, Dataset
import hydra
import os

@hydra(config_path='./configs', config_name='remote_main')
def main(cfg):

    task = Task.init(
        project_name=cfg.clearml.project_name, task_name=cfg.clearml.task_name, 
        output_uri=cfg.clearml.output_uri, task_type=cfg.task_type
    )
    task.set_base_docker(cfg.clearml.base_docker_image)
    # duplicate task, reset state, and send to remote clearml agent, exit process after this
    task.execute_remotely(queue_name=cfg.clearml.queue_name, clone=False, exit_process=True)

    from .train import train_pipeline

    if cfg.task_type == 'training':

        dataset_splits = ['train', 'dev']
        for split in dataset_splits:
            clearml_dataset = Dataset.get(dataset_id=cfg.dataset[f'{split}_clearml_id'])
            split_dir = clearml_dataset.get_local_copy()
            cfg.dataset[f'{split}_manifest_path'] = os.path.join(split_dir, cfg.dataset[f'{split}_manifest_name'])

        if cfg.dataset.get('test_clearml_id', None):
            clearml_dataset = Dataset.get(dataset_id=cfg.dataset['test_clearml_id'])
            test_dir = clearml_dataset.get_local_copy()
            cfg.dataset[f'test_manifest_path'] = os.path.join(test_dir, cfg.dataset['test_manifest_name'])

    train_pipeline(cfg)

if __name__ == '__main__':
    main()