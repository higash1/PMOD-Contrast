# PMOD-Net: point-cloud-map-based metric scale obstacle detection by using a monocular camera

## 依存

- NVIDIA-Driver `>=418.81.07`
- Docker `>=19.03`
- NVIDIA-Docker2

## Dockerイメージ

- pull
    ```bash
    docker pull shikishimatasakilab/pmod
    ```

- build
    ```bash
    docker pull pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
    ```
    ```bash
    ./docker/build.sh -i pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
    ```

### Optunaを使用する場合

- 次のコマンドでDockerイメージをプル.
    ```bash
    docker pull mysql
    ```

## データセットの準備

### KITTI-360

1. KITTI-360データセットを"[h5_kitti360](https://github.com/shikishima-TasakiLab/h5_kitti360)"を用いてHDF5に変換する.
1. データローダの設定として, 学習時は`./config/kitti360-5class.json`を, Validation・評価時は`./config/kitti360-5class-ins.json`を使用する.

### Other Datasets

1. [このページ](OTHER_DATASETS.md)を参照.

### 静的カメラ画像の準備

1. [このリンク](https://github.com/higash1/IOPaint)

## Dockerコンテナの起動

1. 次のコマンドでDockerコンテナを起動する.
    ```bash
    ./docker/run.sh -d path/of/the/dataset/dir
    ```
    ```text
    Usage: run.sh [OPTIONS...]
    OPTIONS:
        -h, --help          Show this help
        -i, --gpu-id ID     Specify the ID of the GPU
        -d, --dataset-dir   Specify the directory where datasets are stored
    ```

### hdf5ファイルに静的カメラ画像をセット

1. 静的カメラ画像をhdf5ファイルにセットする
    例: /workspace/pmod/preprocess/scripts/set_iopaintimg.sh
    ```bash
    python /workspace/pmod/preprocess/utils/set_iopaintimg.py
        -hdf5 /path/of/the/dataset1.hdf5
        -c /path/of/the/config.json
        -d /path/of/the/static_camera_img_dir
    ```
    ```text
    usage: set_iopaintimg.py
         -hdf5 PATH of training HDF5 dataset.
         -c PATH of JSON file of dataloader config for training.
         -d PATH of static camera image directory.
    ```

## 学習

1. 次のコマンドで学習を開始する.
    ```bash
    python train.py -t TAG -tdc path/of/the/config.json \
      -td path/of/the/dataset1.hdf5 [path/of/the/dataset1.hdf5 ...] \
      -vdc path/of/the/config.json \
      -op /workspace/pmod/config/optim-params-moldvec.yaml \
      --shared-weights \
      --mold-vector \
      -b BATCH_SIZE

      自己位置誤差なしの学習、小テストのオプション:
      -ppa --tr-error-range 0.0 0.0 0.0 --rot_error-range 0.0
    ```
    ```text
    usage: train.py [-h] -t TAG -tdc PATH [-vdc PATH] [-bs BLOCK_SIZE]
                    -td PATH [PATH ...] [-vd [PATH [PATH ...]]] [--epochs EPOCHS]
                    [--epoch-start-count EPOCH_START_COUNT]
                    [--steps-per-epoch STEPS_PER_EPOCH] [-ppa]
                    [--tr-error-range TR_ERROR_RANGE TR_ERROR_RANGE TR_ERROR_RANGE]
                    [--rot-error-range ROT_ERROR_RANGE] [-ae] [-edc PATH]
                    [-ed [PATH [PATH ...]]] [--thresholds THRESHOLDS [THRESHOLDS ...]]
                    [--seed SEED] [--augmentation] [-b BATCH_SIZE] [--resume PATH] [-amp]
                    [--clip-max-norm CLIP_MAX_NORM] [-op PATH]
                    [-o {adam,sgd,adabelief}] [-lp {lambda,step,plateau,cos,clr}]
                    [--filter_radius FILTER_RADIUS]
                    [--l1 L1] [--seg-ce SEG_CE] [--seg-ce-aux1 SEG_CE_AUX1]
                    [--seg-ce-aux2 SEG_CE_AUX2] [--detect-anomaly]

                    Contrast:
                      Network
                         --shared-weights --mold-vector
                      loss
                         [--contrastive CONTRASTIVE]

    optional arguments:
      -h, --help            show this help message and exit

    Training:
      -t TAG, --tag TAG     Training Tag.
      -tdc PATH, --train-dl-config PATH
                            PATH of JSON file of dataloader config for training.
      -vdc PATH, --val-dl-config PATH
                            PATH of JSON file of dataloader config for validation.
                            If not specified, the same file as "--train-dl-config" will be used.
      -bs BLOCK_SIZE, --block-size BLOCK_SIZE
                            Block size of dataset.
      -td PATH [PATH ...], --train-data PATH [PATH ...]
                            PATH of training HDF5 datasets.
      -vd [PATH [PATH ...]], --val-data [PATH [PATH ...]]
                            PATH of validation HDF5 datasets. If not specified,
                            the same files as "--train-data" will be used.
      --epochs EPOCHS       Epochs
      --epoch-start-count EPOCH_START_COUNT
                            The starting epoch count
      --steps-per-epoch STEPS_PER_EPOCH
                            Number of steps per epoch. If it is greater than the total number
                            of datasets, then the total number of datasets is used.
      -ppa, --projected-position-augmentation
                            Unuse Projected Positiion Augmentation
      --tr-error-range TR_ERROR_RANGE TR_ERROR_RANGE TR_ERROR_RANGE
                            Translation Error Range [m].
      --rot-error-range ROT_ERROR_RANGE
                            Rotation Error Range [deg].
      -ae, --auto-evaluation
                            Auto Evaluation.
      -edc PATH, --eval-dl-config PATH
                            PATH of JSON file of dataloader config.
      -ed [PATH [PATH ...]], --eval-data [PATH [PATH ...]]
                            PATH of evaluation HDF5 datasets.
      --thresholds THRESHOLDS [THRESHOLDS ...]
                            Thresholds of depth.
      --seed SEED           Random seed.

      --augmentation        Brightness augmentation.


    Network:
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            Batch Size
      --resume PATH         PATH of checkpoint(.pth).
      -amp, --amp           Use AMP.

      Contrast:
        --shared-weights    Weight sharing with PMOD-Net.
        --mold-vector       Shape features into GT Segmentation images.

    Optimizer:
      --clip-max-norm CLIP_MAX_NORM
                            max_norm for clip_grad_norm.
      -op PATH, --optim-params PATH
                            PATH of YAML file of optimizer params.
      -o {adam,sgd,adabelief}, --optimizer {adam,sgd,adabelief}
                            Optimizer
      -lp {lambda,step,plateau,cos,clr}, --lr-policy {lambda,step,plateau,cos,clr}
                            Learning rate policy.
    
    Inmap:
      --filter_radius FILTER_RADIUS
                         Inmap filtering radius settings default:0 CMRNet_filtering:5                        

    Loss:
      --l1 L1               Weight of L1 loss.
      --seg-ce SEG_CE       Weight of Segmentation CrossEntropy Loss.
      --seg-ce-aux1 SEG_CE_AUX1
                            Weight of Segmentation Aux1 CrosEntropy Loss.
      --seg-ce-aux2 SEG_CE_AUX2
                            Weight of Segmentation Aux2 CrosEntropy Loss.
      
      Contrast:
        --contrastive       Weight of Triplet loss.                      

    Debug:
      --detect-anomaly      AnomalyMode
    ```

2. チェックポイントは"./checkpoints"ディレクトリに保存される.

    ```text
    checkpoints/
    　├ YYYYMMDDThhmmss-TAG/
    　│　├ config.yaml
    　│　├ 00001_PMOD.pth
    　│　├ :
    　│　├ :
    　│　├ EPOCH_PMOD.pth
    　│　└ validation.xlsx
    ```

## 評価

1. 次のコマンドで評価を行う.
    PMOD-Contrastの場合
    ```bash
    python evaluate_Contrast.py -t TAG -cp path/of/the/checkpoint.pth \
      -edc path/of/the/config.json \
      -ed path/of/the/dataset1.hdf5 [path/of/the/dataset2.hdf5 ...]
    ```
    PMOD-Netの場合
    ```bash
    python evaluate.py -t TAG -cp path/of/the/checkpoint.pth \
      -edc path/of/the/config.json \
      -ed path/of/the/dataset1.hdf5 [path/of/the/dataset2.hdf5 ...]
    ```
    ```text
    usage: evaluate.py [-h] -t TAG -cp PATH -edc PATH [-bs BLOCK_SIZE]
                       -ed PATH [PATH ...] [--train-config PATH]
                       [--thresholds THRESHOLDS [THRESHOLDS ...]]
                       [--seed SEED] [-b BATCH_SIZE]

    optional arguments:
      -h, --help            show this help message and exit

    Evaluation:
      -t TAG, --tag TAG     Evaluation Tag.
      -cp PATH, --check-point PATH
                            PATH of checkpoint.
      -edc PATH, --eval-dl-config PATH
                            PATH of JSON file of dataloader config.
      -bs BLOCK_SIZE, --block-size BLOCK_SIZE
                            Block size of dataset.
      -ed PATH [PATH ...], --eval-data PATH [PATH ...]
                            PATH of evaluation HDF5 datasets.
      --nomap               No map input.
      --tr-error-range TR_ERROR_RANGE TR_ERROR_RANGE TR_ERROR_RANGE
                            Translation Error Range [m]. This is used when the
                            data do not contain poses.
      --rot-error-range ROT_ERROR_RANGE
                            Rotation Error Range [deg]. This is used when the data
                            do not contain poses.
      --train-config PATH   PATH of "config.yaml"
      --thresholds THRESHOLDS [THRESHOLDS ...]
                            Thresholds of depth.
      --seed SEED           Random seed.

    Network:
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            Batch Size
    ```

1. 評価結果は"./results"ディレクトリに保存される.

    ```text
    results/
    　├ YYYYMMDDThhmmss-TRAINTAG/
    　│　├ YYYYMMDDThhmmss-TAG/
    　│　│　├ config.yaml
    　│　│　├ data.hdf5
    　│　│　└ result.xlsx
    ```

## data.hdf5 &rarr; ビデオ (.avi)

1. 評価結果に含まれる "data.hdf5" は次のコマンドでビデオ(.avi)に変換する.
    ```bash
    python data2avi.py -i path/of/the/data.hdf5
    ```
    ```text
    usage: data2avi.py [-h] -i PATH [-o PATH] [-r]

    optional arguments:
      -h, --help            show this help message and exit
      -i PATH, --input PATH
                            Input path.
      -o PATH, --output PATH
                            Output path. Default is "[input dir]/data.avi"
      -r, --raw             Use raw codec
    ```

1. 変換したビデオは入力のHDF5ファイルと同じディレクトリに保存される.

## data.hdf5 -> 点群

1. Generate backprojected point cloud from data.hdf5
    ```bash
    python /workspace/pmod/data2backprojection.py \
      -hdf5 /path/of/the/data.hdf5 \
      -n TAG \
      -c /path/of/the/config.json \
      --dataset NAME
    ```
    ```text
    usage: data2backprojection.py -hdf5 PATH -n TAG -c PATH --dataset NAME [-rm] [-pc]
      -hdf5 PATH
                Input PATH
      -c PATH
                Config PATH
      -n TAG 
                Output directory name
      --dataset Name
                (carla or kitti or real)
      -rm
                remove corresponiding point cloud by label
      -pc
                not create point cloud
    ```

    ※ カメラの内部パラメータを持つhdf5ファイルパスをdata2backprojection.py内に記述する必要があります。

## data.hdf5 -> 再評価

1. 15classで評価したものを2クラスなどで再評価する
    ```bash
    python /workspace/pmod/data2reevaluate.py \
      -i path/of/the/data.hdf5 \
      -c /path/of/the/config_2class.json
    ```
    ```text
    usage: data2reevaluate.py -i PATH -c PATH
      -i PATH, --input PATH
                      Input path.
      -c PATH
                      Config path. (differnt class)
    ```

    出力
    ```text
      results/ 
       YYYYMMDDThhmmss-TRAINTAG
          ├ YYYYMMDDThhmmss-TAG/
            ├ reresults.xlsx
            ├ ...
    ```

## xlsxs -> xlsx

1. 評価後のresults.xlsxをresults_all.xlsxにまとめる
    ```bash
    python result2all.py -id /dir/of/the/results
    ```
    ```text
    usage: result2all.py -id DIR_PATH [-re]
      -id DIR_PATH 
                      Results dir (Two levels above results.xlsx -> YYYYMMDDThhmmss-TRAINTAG)
      -re 
                      Summarizing the results in reresults.xlsx
    ```
    出力
    ```text
      results/ 
       YYYYMMDDThhmmss-TRAINTAG
          ├ results_all.xlsx or reresults_all.xlsx
          ├ YYYYMMDDThhmmss-TAG/
            ├ ...
    ```

## チェックポイント (.pth) &rarr; Torch Script model (.pt)

学習したモデルをLibTorchで使用する場合などに実行する.

1. 次のコマンドでチェックポイント(.pth)をTorch Script model (.pt)に変換する.
    ```bash
    python ckpt2tsm.py -c path/of/the/checkpoint.pth
    ```
    ```text
    usage: ckpt2tsm.py [-h] [-c PATH] [-x WIDTH] [-y HEIGHT]

    optional arguments:
      -h, --help            show this help message and exit
      -c PATH, --check-point PATH
                            Path of checkpoint file.
      -x WIDTH, --width WIDTH
                            Width of input images.
      -y HEIGHT, --height HEIGHT
                            Height of input images.
    ```

1. 変換したTorch Script modelは入力したチェックポイントと同じディレクトリに保存される.

## Training with Optuna

1. MySQLのDockerコンテナを別のターミナルで起動する.
    ```bash
    ./optuna/run-mysql.sh
    ```

1. 次のコマンドで学習を開始する.
    ```bash
    python optuna_train.py -t TAG -tdc path/of/the/config.json \
      -td path/of/the/dataset1.hdf5 [path/of/the/dataset2.hdf5 ...] -bs BLOCK_SIZE
    ```
    ```text
    usage: optuna_train.py [-h] [--seed SEED] [-n N_TRIALS] -t TAG [-H HOST]
                           [-s {tpe,grid,random,cmaes}] -tdc PATH [-vdc PATH]
                           [-bs BLOCK_SIZE] -td PATH [PATH ...] [-vd [PATH [PATH ...]]]
                           [--epochs EPOCHS] [--epoch-start-count EPOCH_START_COUNT]
                           [--steps-per-epoch STEPS_PER_EPOCH] [-ppa]
                          [--tr-error-range TR_ERROR_RANGE]
                          [--rot-error-range ROT_ERROR_RANGE]
                          [-b BATCH_SIZE] [--resume PATH] [-amp]
                          [--clip-max-norm CLIP_MAX_NORM] [-op PATH]
                          [-o {adam,sgd,adabelief}] [-lp {lambda,step,plateau,cos,clr}]
                          [--l1 L1] [--seg-ce SEG_CE] [--seg-ce-aux1 SEG_CE_AUX1]
                          [--seg-ce-aux2 SEG_CE_AUX2] {multi} ...

    positional arguments:
      {multi}
        multi               Multi Objective Trial

    optional arguments:
      -h, --help            show this help message and exit

    Optuna:
      --seed SEED           Seed for random number generator.
      -n N_TRIALS, --n-trials N_TRIALS
                            Number of trials.
      -t TAG, --tag TAG     Optuna training tag.
      -H HOST, --host HOST  When using a MySQL server, specify the hostname.
      -s {tpe,grid,random,cmaes}, --sampler {tpe,grid,random,cmaes}
                            Optuna sampler.

    Training:
      -tdc PATH, --train-dl-config PATH
                            PATH of JSON file of dataloader config for training.
      -vdc PATH, --val-dl-config PATH
                            PATH of JSON file of dataloader config for validation.
                            If not specified, the same file as "--train-dl-config"
                            will be used.
      -bs BLOCK_SIZE, --block-size BLOCK_SIZE
                            Block size of dataset.
      -td PATH [PATH ...], --train-data PATH [PATH ...]
                            PATH of training HDF5 datasets.
      -vd [PATH [PATH ...]], --val-data [PATH [PATH ...]]
                            PATH of validation HDF5 datasets. If not specified,
                            the same files as "--train-data" will be used.
      --epochs EPOCHS       Epochs
      --epoch-start-count EPOCH_START_COUNT
                            The starting epoch count
      --steps-per-epoch STEPS_PER_EPOCH
                            Number of steps per epoch. If it is greater than the
                            total number of datasets, then the total number of
                            datasets is used.
      -ppa, --projected-position-augmentation
                            Unuse Projected Positiion Augmentation
      --tr-error-range TR_ERROR_RANGE
                            Translation Error Range [m].
      --rot-error-range ROT_ERROR_RANGE
                            Rotation Error Range [deg].

    Network:
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            Batch Size
      --resume PATH         PATH of checkpoint(.pth).
      -amp, --amp           Use AMP.

    Optimizer:
      --clip-max-norm CLIP_MAX_NORM
                            max_norm for clip_grad_norm.
      -op PATH, --optim-params PATH
                            PATH of YAML file of optimizer params.
      -o {adam,sgd,adabelief}, --optimizer {adam,sgd,adabelief}
                            Optimizer
      -lp {lambda,step,plateau,cos,clr}, --lr-policy {lambda,step,plateau,cos,clr}
                            Learning rate policy.

    Loss:
      --l1 L1               Weight of L1 loss.
      --seg-ce SEG_CE       Weight of Segmentation CrossEntropy Loss.
      --seg-ce-aux1 SEG_CE_AUX1
                            Weight of Segmentation Aux1 CrosEntropy Loss.
      --seg-ce-aux2 SEG_CE_AUX2
                            Weight of Segmentation Aux2 CrosEntropy Loss.
    ```

1. 他のマシンを用いて並列に学習する場合は, `-H`オプションを用いてMySQLを実行しているサーバーを指定する.

## 主な参考文献

```bibtex
@article{pmodnet,
    author = {Junya Shikishima and Keisuke Urasaki and Tsuyoshi Tasaki},
    title = {PMOD-Net: point-cloud-map-based metric scale obstacle detection by using a monocular camera},
    journal = {Advanced Robotics},
    pages = {1-9},
    year  = {2022},
    publisher = {Taylor & Francis},
    doi = {10.1080/01691864.2022.2153080},
    URL = {
        https://doi.org/10.1080/01691864.2022.2153080
    },
    eprint = {
        https://doi.org/10.1080/01691864.2022.2153080
    }
}
```
