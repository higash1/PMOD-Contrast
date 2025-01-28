import csv
import os
import numpy as np
import argparse

from tqdm import tqdm
from typing import List
import matplotlib.pyplot as plt
import random

# constants
MEGAROVER3_WIDTH: float = 0.20

ROBOBY_WIDTH: float = 0.15 / 2.
MEGAROVER4_WIDTH: float = 0.35 / 2.
FOXDALL_WIDTH: float = 0.30 / 2.
MAN_WIDTH: float = 0.50 / 2.
MAN_LENGTH: float = 0.30 / 2.

# put the obstacle location x, y = 2.0, -0.1
obstacle_location = np.array([2.0, -0.1])


# voxel 書きたかっただけ
#### voxel setting ####

#                    ul2 -- ################  -- ur2
#                  ul1     / |            /|
#                      \  /  |           / |  -- ur1
#                        ################  |
#                     -- |  /           |  |    -- br2
#                    /   | /            | /
#                 bl2    |/             |/
#                bl1  -- ################  -- br1
# 
#  bl1: bottom left 1  bl2: bottom left 2  ul1: upper left 1  ul2: upper left 2
#  br1: bottom right 1 br2: bottom right 2 ur1: upper right 1 ur2: upper right 2


class RouteEvaluate:
    def __init__(self, args):
        origin_path = args.origin_path
        predict_path = args.predict_path
        self.tag = args.tag
        self.width = None
        self.length = None
        
        # define width for the voxel
        if args.key == 'roboby':
            self.width = ROBOBY_WIDTH
        elif args.key == 'megarover4':
            self.width = MEGAROVER4_WIDTH
        elif args.key == 'foxdall':
            self.width = FOXDALL_WIDTH
        elif args.key == 'man':
            self.width = MAN_WIDTH
            self.length = MAN_LENGTH
            
        # 2D box setting xy平面のみでぶつかっているかを判定するため
        self.obstacle_box = np.zeros((1, 4, 2))
        self.megarover_box = np.zeros((1, 4, 2))
        # set obstacle box & initial megarover box
        self.box_generate()

        origin_header, origin_data = self.read_csv(origin_path)
        predict_header, predict_data = self.read_csv(predict_path)

        origin_array = self.data2array(origin_data)
        predict_array = self.data2array(predict_data)
        # xyzrpy -> xyz
        origin_array = origin_array[:,:3]

        # origin_arrayから初期値を削除 & initial position 取得 predict_arrayは初期値を参照しないため
        initial_position: np.ndarray = origin_array[0]
        origin_array: np.ndarray = origin_array[1:]
        
        astar_route_list:List[np.ndarray] = []
        match_index: np.ndarray = np.where(np.all(predict_array == origin_array[-1], axis=1))[0]
        # if match_index.shape[0] != 0:
        # 最後の経路
        astar_route: np.ndarray = predict_array[match_index[-2] + 1: match_index[-1] + 1]
        astar_route_list.append(astar_route)
        
        inv_match_index:np.ndarray = match_index[::-1] # 逆順にする
        for idx in tqdm(range(match_index.shape[0] - 1), desc='astar route search'):
            # if idx == 0:
            #     continue;
            start_idx = inv_match_index[idx + 1]
            end_idx = inv_match_index[idx]
            astar_route = predict_array[start_idx + 1: end_idx + 1]
            if astar_route.shape[0] != astar_route_list[-1].shape[0]:
                astar_route_list.append(astar_route)
            else:
                for row in zip(astar_route, astar_route_list[-1]):
                    if not np.all(row[0] == row[1]):
                        astar_route_list.append(astar_route)
                        continue;
                    else:
                        continue;
        
        # origin_arrayと同じ経路をastar_route_listから削除
        for num, astar_route in enumerate(astar_route_list):
            if astar_route.shape[0] == origin_array.shape[0]:
                astar_route_list.pop(num)
        
        # すべての回避経路をcsvに保存 astar_route_list 最後から
        collision_num = 0
        route_distances: List[int, np.ndarray, float, float, float] = []
        for num, astar_route in enumerate(astar_route_list[::-1]):
            if not self.box_collision_check(astar_route):
                origin_distance, predict_distance = self.route_distance_calculate(origin_array, astar_route)
                distance_diff = predict_distance - origin_distance
                route_distances.append((num, astar_route, origin_distance, predict_distance, distance_diff))
            else:
                collision_num += 1

        # random 10
        # sampled_routes = random.sample(route_distances, 5)
            
            # top 5
            # route_distances.sort(key=lambda x: x[4])
            # top_routes = route_distances[:5]
        
        for num, route, origin_distance, predict_distance, _ in sorted(route_distances):
            self.save_csv(num, origin_distance, predict_distance)
            self.plot(num, origin_array, route)
        
        print(f'collision_num: {collision_num}/ {len(astar_route_list)}')
        # average diff distance in sampled routes
        average_sampled_predict_distance = np.mean([route[3] for route in route_distances])
        average_sampled_origin_distance = np.mean([route[2] for route in route_distances])

        with open(f'route_evaluate_{self.tag}.csv', 'a') as f:
            writer = csv.writer(f)
            # 改行
            writer.writerow([])
            # average diff distance in sampled routes
            writer.writerow([f'sample average','origin_distance','predict_distance','diff'])
            writer.writerow(['',average_sampled_origin_distance,average_sampled_predict_distance,average_sampled_predict_distance - average_sampled_origin_distance])

            writer.writerow([])
            # collision num
            writer.writerow(['collision_num', f'{collision_num}' '/' f'{len(astar_route_list)}' f'{100 - (collision_num / len(astar_route_list) * 100): 6f} %'])
                

    def box_generate(self):
        ### box generate
        if self.length is not None:
            self.obstacle_box[0, 0] = (-self.length, -self.width)
            self.obstacle_box[0, 3] = (self.length, -self.width)
            self.obstacle_box[0, 1] = (-self.length, self.width)
            self.obstacle_box[0, 2] = (self.length, self.width)
        else:
            # bottom left
            self.obstacle_box[0, 0] = (-self.width, -self.width)
            # bottom right
            self.obstacle_box[0, 3] = (self.width, -self.width)
            # upper left
            self.obstacle_box[0, 1] = (-self.width, self.width)
            # upper right
            self.obstacle_box[0, 2] = (self.width, self.width)
        
        # set location
        self.obstacle_box[0] += obstacle_location
        
        # megarover
        # bottom left
        self.megarover_box[0, 0] = (-MEGAROVER3_WIDTH, -MEGAROVER3_WIDTH)
        # bottom right
        self.megarover_box[0, 3] = (MEGAROVER3_WIDTH, -MEGAROVER3_WIDTH)
        # upper left
        self.megarover_box[0, 1] = (-MEGAROVER3_WIDTH, MEGAROVER3_WIDTH)
        # upper right
        self.megarover_box[0, 2] = (MEGAROVER3_WIDTH, MEGAROVER3_WIDTH)
        
    def box_collision_check(self, route_ex: np.ndarray):
        self.megarover_boxes_list: List[np.ndarray] = []
        megarover_box = np.zeros((1, 4, 2))
        for position in route_ex:
            megarover_box[0] = self.megarover_box[0] + position[:2]
            if self.check_collision(megarover_box=megarover_box):
                return True
            self.megarover_boxes_list.append(megarover_box.copy())
        return False  
    
    def check_collision(self, megarover_box: np.ndarray):
        # メガローバーの各頂点が障害物の内部にあるかどうかを確認
        for megarover_corner in megarover_box[0]:
            if (self.obstacle_box[0, 0, 0] <= megarover_corner[0] <= self.obstacle_box[0, 2, 0] and
                self.obstacle_box[0, 0, 1] <= megarover_corner[1] <= self.obstacle_box[0, 2, 1]):
                return True
        
        # 障害物の各頂点がメガローバーの内部にあるかどうかを確認
        for obstacle_corner in self.obstacle_box[0]:
            if (megarover_box[0, 0, 0] <= obstacle_corner[0] <= megarover_box[0, 2, 0] and
                megarover_box[0, 0, 1] <= obstacle_corner[1] <= megarover_box[0, 2, 1]):
                return True
        
        return False

    def read_csv(self, path):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            data = [row for row in reader]
        return header, data
    
    def data2array(self, data):
        float_data = [[float(d) for d in row] for row in data]
        return np.array(float_data)
    
    def plot(self, route_num: int, origin_data: np.ndarray, predict_data: np.ndarray=None, demo=False):
        plt.plot(- origin_data[:, 1], origin_data[:, 0], label='origin')
        plt.plot(- predict_data[:, 1], predict_data[:, 0], label='predict')
        
        for box in self.obstacle_box:
            box = np.append(box, [box[0]], axis=0)
            plt.plot(- box[:, 1], box[:, 0], 'r', label='obstacle')
        # for boxes in self.megarover_boxes_list:
        #     for box in boxes:
        #         box = np.append(box, [box[0]], axis=0)
        #         plt.plot(- box[:, 1], box[:, 0], 'g')
        
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'route_evaluate_{self.tag}_{route_num}.png')
        if demo:
            plt.show()
        plt.close()
        
    def save_csv(self, num,origin_distance, predict_distance):
        with open(f'route_evaluate_{self.tag}.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([f'route {num}','origin_distance','predict_distance','diff'])
            writer.writerow(['',origin_distance,predict_distance,predict_distance - origin_distance])
        
    def route_distance_calculate(self, origin_data: np.ndarray, predict_data: np.ndarray):
        origin_distances = np.sqrt(np.sum(np.diff(origin_data, axis=0) ** 2, axis=1))
        predict_distances = np.sqrt(np.sum(np.diff(predict_data, axis=0) ** 2, axis=1))
        
        origin_total_distance = np.sum(origin_distances)
        predict_total_distance = np.sum(predict_distances)
        
        return origin_total_distance, predict_total_distance

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_path', type=str, default='origin.csv')
    parser.add_argument('--predict_path', type=str, default='predict.csv')
    parser.add_argument('-t', '--tag', type=str, default='default')
    parser.add_argument('--key', type=str, required=True, default=None, help='roboby or megarover4 or foxdall or man')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()
    route_evaluate = RouteEvaluate(args)
        