import argparse
import os
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

ALL = 'All'
METRICS = 'Metrics'
EACH_CLASS = 'Each_class'
EACH_DIST = 'Each_dist'

DIST_CLASS = 14

DEFAULT_OUTPUT = 'result_all.xlsx'
REEVALUATE_OUTPUT = 'reresult_all.xlsx'

DEFAULT_PATH = 'result.xlsx'
REEVALUATE_PATH = 'reresult.xlsx'

def parse_args() -> Dict[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-id', '--inputsdir',
        type=str, metavar='PATH', required=True,
        help='Input Directory Path'
    )
    parser.add_argument(
        '-o', '--output',
        type=str, metavar='PATH', default=None,
        help='Output path. Default is "[input dir]/result_all.xlsx"'
    )
    parser.add_argument(
        '-re', '--reevaluate',
        action='store_true',
        help='Reevaluate the result'
    )
    args = vars(parser.parse_args())

    if isinstance(args['output'], str):
        if os.path.isdir(os.path.dirname(args['output'])) is False:
            raise NotADirectoryError(os.path.dirname(args['output']))
    else:
        output_dir: str = args['inputsdir']
        if args['reevaluate']:
            args['output'] = os.path.join(output_dir, REEVALUATE_OUTPUT)
        else:
            args['output'] = os.path.join(output_dir, DEFAULT_OUTPUT)
    return args


def read_xlsx(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path, index_col=None)

def remove_str_only_rows(np_array: np.ndarray) -> np.ndarray:
    # 各行がすべてstr型のみで構成されている場合、その行を除去
    mask = np.array([all(isinstance(item, str) for item in row) for row in np_array])
    return np_array[~mask]


def main(args: Dict[str, str]):
    inputs_dir: str = args['inputsdir']
    output_path: str = args['output']

    if os.path.exists(output_path):
        os.remove(output_path)
    
    sheet_setting_dict: Dict[str, Tuple[str, str]] = {}
    tmp_result_dict: Dict[str, pd.DataFrame] = {}
    
    result_dirs:List[str] = sorted(os.listdir(inputs_dir))

    # initial setting
    if args['reevaluate']:
        result_df = read_xlsx(os.path.join(inputs_dir, result_dirs[0], REEVALUATE_PATH))
    else:
        result_df = read_xlsx(os.path.join(inputs_dir, result_dirs[0], DEFAULT_PATH))
    # result_df:pd.DataFrame = read_xlsx(os.path.join(inputs_dir, result_dirs[0], 'result.xlsx'))
    metrics = result_df.iloc[:, 0]
    all = result_df.columns[1]
    
    each_dists = result_df.columns[-DIST_CLASS:]
    each_classes = result_df.columns[2:len(result_df.columns)-DIST_CLASS]

    # sheet setting 
    sheet_setting_dict[METRICS] = metrics
    sheet_setting_dict[ALL] = all
    sheet_setting_dict[EACH_CLASS] = each_classes
    sheet_setting_dict[EACH_DIST] = each_dists
    
    metrics_list:list = sheet_setting_dict[METRICS].to_list()
    
    for result_dir in tqdm(result_dirs):
        current_result_dir:str = os.path.join(inputs_dir, result_dir)
        if args['reevaluate']:
            if os.path.exists(os.path.join(current_result_dir, REEVALUATE_PATH)):
                result_df:pd.DataFrame = read_xlsx(os.path.join(current_result_dir, REEVALUATE_PATH))
            else:
                print(f"{REEVALUATE_PATH} not found in {current_result_dir}")
                continue
        else:
            if os.path.exists(os.path.join(current_result_dir, DEFAULT_PATH)):
                result_df:pd.DataFrame = read_xlsx(os.path.join(current_result_dir, DEFAULT_PATH))
            else:
                print(f"{DEFAULT_PATH} not found in {current_result_dir}")
                continue
        if len(result_dir.split('-')) == 1:
            tmp_result_dict[result_dir] = {}
        else:
            tmp_result_dict[result_dir.split('-')[1]] = {}
        # if os.path.exists(os.path.join(current_result_dir, 'result.xlsx')):
            # result_df:pd.DataFrame = read_xlsx(os.path.join(current_result_dir, 'result.xlsx'))

        # data extraction
        if len(result_dir.split('-')) == 1:
            tmp_result_dict[result_dir][ALL] = result_df.iloc[:, 1]
            tmp_result_dict[result_dir][EACH_CLASS] = result_df.iloc[:, 2:len(result_df.columns)-DIST_CLASS]
            tmp_result_dict[result_dir][EACH_DIST] = result_df.iloc[:, -DIST_CLASS:]
        else:
            tmp_result_dict[result_dir.split('-')[1]][ALL] = result_df.iloc[:, 1]
            tmp_result_dict[result_dir.split('-')[1]][EACH_CLASS] = result_df.iloc[:, 2:len(result_df.columns)-DIST_CLASS]
            tmp_result_dict[result_dir.split('-')[1]][EACH_DIST] = result_df.iloc[:, -DIST_CLASS:]
    
    all_df = pd.DataFrame({key: value[ALL] for key, value in tmp_result_dict.items()})
    
    # each class result
    class_result_list = []
    for key, value in tmp_result_dict.items():
        # keyとカラム名を結合して1行目に設定
        header_row = value[EACH_CLASS].columns.insert(0, key)
        class_result_list.append(header_row)
        
        # 各結果の値を追加
        for row, metric in zip(value[EACH_CLASS].values, metrics_list):
            row_list = row.tolist()
            row_list.insert(0, metric)
            class_result_list.append(row_list)
        
        # 空白行を追加
        class_result_list.append([''] * len(header_row))

    # 結果をデータフレームに変換
    class_result_df = pd.DataFrame(class_result_list)

    # each dist result
    dist_result_list = []
    for key, value in tmp_result_dict.items():
        # keyとカラム名を結合して1行目に設定
        header_row = value[EACH_DIST].columns.insert(0, key)
        dist_result_list.append(header_row)
        
        # 各結果の値を追加
        for row, metric in zip(value[EACH_DIST].values, metrics_list):
            row_list = row.tolist()
            row_list.insert(0, metric)
            dist_result_list.append(row_list)
        
        # 空白行を追加
        dist_result_list.append([''] * len(header_row))
    
    dist_result_df = pd.DataFrame(dist_result_list)

    # nanに対処しながら全体の平均値を算出する
    np_all_values:np.ndarray = all_df.values.reshape(4, -1).astype(np.float32)
    all_nan_condition = ~np.isnan(np_all_values)
    all_average_value = np.where(all_nan_condition, np_all_values, 0).sum(axis=1) / all_nan_condition.sum(axis=1)
    
    # remove column only str
    np_class_values:np.ndarray = class_result_df.values
    
    rm_class_values:np.ndarray = remove_str_only_rows(np_class_values)
    ini_rm_class_values:np.ndarray = np.delete(rm_class_values, 0,axis=1).reshape(int(rm_class_values.shape[0] / 4), 4, -1).astype(np.float32)
    class_nan_condition = ~np.isnan(ini_rm_class_values)
    class_average_value = np.where(class_nan_condition, ini_rm_class_values, 0).sum(axis=0) / class_nan_condition.sum(axis=0)
    
    # remove column only str
    np_dist_values:np.ndarray = dist_result_df.values
    rm_dist_values:np.ndarray = remove_str_only_rows(np_dist_values)
    ini_rm_dist_values:np.ndarray = np.delete(rm_dist_values, 0,axis=1).reshape(int(rm_class_values.shape[0] / 4), 4, -1).astype(np.float32)
    dist_nan_condition = ~np.isnan(ini_rm_dist_values)
    dist_average_value = np.where(dist_nan_condition, ini_rm_dist_values, 0).sum(axis=0) / dist_nan_condition.sum(axis=0)


    average_values:np.ndarray = np.hstack([all_average_value.reshape(4, -1), class_average_value, dist_average_value])

    average_columns = [sheet_setting_dict[ALL]] + sheet_setting_dict[EACH_CLASS].to_list() + sheet_setting_dict[EACH_DIST].to_list()
    
    average_df = pd.DataFrame(average_values, columns=average_columns)
    average_df.insert(0, 'Metric', sheet_setting_dict[METRICS])
    
    all_df.insert(0, 'Metric', sheet_setting_dict[METRICS])
    
    # Excelファイルに書き込み
    with pd.ExcelWriter(output_path) as writer:
        average_df.to_excel(writer, sheet_name='Average', index=False)
        pd.DataFrame(all_df).to_excel(writer, sheet_name='All', index=False)
        pd.DataFrame(class_result_df).to_excel(writer, sheet_name='Each_Class', index=False, columns=None, header=None)
        pd.DataFrame(dist_result_df).to_excel(writer, sheet_name='Each_Dist', index=False, columns=None, header=None)

if __name__ == '__main__':
    args = parse_args()
    main(args)
