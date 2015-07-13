#!/usr/bin/python
#-*- encoding: utf-8 -*-
"""
KMR (Kandori-Mailath-Rob) Model
Copyright (c) 2015 @myuuuuun
https://github.com/myuuuuun/KMR

Released under the MIT license.
"""
# エラー処理の類はまた今度
# 無駄が多い部分があるものの、それもまた今度

import numpy as np
import scipy as sc
import quantecon as qe
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np..set_printoptions(threshold=numpy.nan)

# transition_matrixの各行（or列）と[行動0の人数, 行動1の人数, 行動2の人数,...]を対応付ける
def make_state_players(num_players, num_actions):
    if num_actions == 1:
        return [[num_players]]

    rst = [[i] + li for i in range(num_players+1) for li in make_state_players(num_players-i, num_actions-1)]
    return rst


# 行プレイヤーの利得表を返す
def make_profit_matrix(profits):
    profit_matrix = np.asarray(profits)
    row, col = np.shape(profit_matrix)
    if row != col:
        print("利得表の行列数が一致しません")
        exit(-1)

    return profit_matrix


def best_response(profit_matrix, state, current_action):
    """
    最適反応を返す

    Input
    -----
    profit_matrix: ndarray(int, ndim=2)
        利得行列

    state: ndarray(int, ndim=1)
        行動i(i=0, 1...)をとっているプレイヤーの数の配列

    current_action: scalar(int)
        行動を変更するプレイヤーの、現在の行動

    Output
    ------
    best_action: ndarray(int, ndim=1)
        最適反応の配列（同期待値の行動が複数ある場合は、要素が複数入る）
    """
    num_actions = len(state)
    expected_values = np.zeros(num_actions)

    for action in range(num_actions):
        #print("現在のプレイヤーの行動は {0} 、stateは {1} 、変更先の行動は {2}".format(current_action, state, action))
        for matched_action, num_players in enumerate(state):
            if matched_action == action:
                num_players += 1

            if matched_action == current_action:
                num_players -= 1

            expected_values[action] += num_players * profit_matrix[action][matched_action]
    
    best_action = np.argwhere(expected_values == np.amax(expected_values)).flatten()
    return best_action


def is_best_response(profit_matrix, state, current_action, new_action):
    """
    最適反応であれば 1 /（最適反応の総数）、最適反応でなければ0を返す

    Input
    -----
    profit_matrix: ndarray(int, ndim=2)
        利得行列

    state: ndarray(int, ndim=1)
        行動i(i=0, 1...)をとっているプレイヤーの数の配列

    current_action: scalar(int)
        行動を変更するプレイヤーの、現在の行動

    new_action: scalar(int)
        行動を変更するプレイヤーの、新しい行動

    Output
    ------
    is_best_response: scalar(float)
        最適反応であれば 1 /（最適反応の総数）、最適反応でなければ0
    """
    best_action = best_response(profit_matrix, state, current_action)
    length = len(best_action)

    return 1.0 / length if new_action in best_action else 0


def kmr_markov_matrix(profit_matrix, num_players, num_actions, epsilon, **kwargs):
    """
    KMRの遷移行列を返す（対称行列ゲームだけを考慮）

    Input
    -----
    profits: array_like(int, ndim=2)
        行列を片方のプレイヤーの利得表と見て、対称行列ゲームの遷移行列を返す

    num_players: scalar(int)
        プレイヤーの総人数

    epsilon: scalar(float), 0 <= epsilon <= 1
        プレイヤーが「実験」を行う確率。「実験」を行う場合、最適反応に関係なく、すべての行動の中から等確率で行動を選択する。

    Output
    ------
    transition_matrix: ndarray(float, ndim=2)
        遷移行列
    """
    # 1度何人が行動変更の機会を得るか
    # 暇な時に対応する（現状は逐次改訂のみ）
    num_changable = 1 #kwargs.get('num_changable', 1)
    if num_changable == 'all':
        num_changable = num_players

    num_states = sc.misc.comb(num_players+num_actions-1, num_actions-1, exact=True)
    
    # どの行動をとっているプレイヤーを何人増減させれば、推移できるか
    def make_move_matrix(num_players, num_actions, state_players_list):
        col_list = np.empty((num_states, num_states, num_actions))
        col_list[:] = state_players_list
        row_list = np.swapaxes(col_list, 0, 1)
        return row_list - col_list


    state_players_list = make_state_players(num_players, num_actions)
    move_matrix = make_move_matrix(num_players, num_actions, state_players_list)

    # 状態遷移に最低何人の人が行動を変更する必要があるか
    move_costs = np.zeros((num_states, num_states), dtype=int)
    for i in range(num_states):
        for j in range(num_states):
            move_players = move_matrix[i][j]
            move_costs[i][j] = np.sum(np.fabs(move_players)) / 2

    # is_movable: 推移行列の各要素が推移可能か
    # 一度に行動を変更できる人数以上に動かさなければ実現できない組み合わせはダメ
    not_movable = np.where(move_costs > num_changable)
    is_movable = np.ones((num_states, num_states), dtype=bool)
    is_movable[not_movable[0:2]] = False

    # 遷移行列を生成
    transition_matrix = np.zeros((num_states, num_states), dtype=float)
    for i in range(num_states):
        for j in range(num_states):
            if is_movable[i][j]:
                move_list = move_matrix[i][j]
                current_state_players = state_players_list[i]

                if 1 in move_list:
                    move_from = np.where(move_list == 1)[0][0]
                    move_to = np.where(move_list == -1)[0][0]
                    transition_matrix[i][j] = (current_state_players[move_from] / num_players) * \
                                                ((1-epsilon) * is_best_response(profit_matrix, current_state_players, move_from, move_to) + epsilon / num_actions)

                else:
                    total = 0
                    for move in range(num_actions):
                        total += (current_state_players[move] / num_players) * \
                                    ((1-epsilon) * is_best_response(profit_matrix, current_state_players, move, move) + epsilon / num_actions)

                    transition_matrix[i][j] = total

    return transition_matrix


class KMR(object):
    """
    Class representing the KMR dynamics.
    """
    def __init__(self, profits, num_players, epsilon):
        self.num_players = num_players
        self.num_actions = 2 if isinstance(profits, int) else len(profits)
        self.num_states = sc.misc.comb(self.num_players + self.num_actions - 1, self.num_actions - 1, exact=True)
        self.state_players = make_state_players(num_players, self.num_actions)
        self.profit_matrix = make_profit_matrix(profits)
        self.transition_matrix = kmr_markov_matrix(profits, num_players, self.num_actions, epsilon)
        self.mc = qe.MarkovChain(self.transition_matrix)


    # 遷移行列の列番号（行番号）から、そのstateにおける各行動の人数のリストを返す
    # 番号がリストで複数与えられたら、人数のリストのリストを返す
    def from_stateindex_to_stateplayersnum(self, stateindex):
        if isinstance(stateindex, int):
            return self.state_players[stateindex]
        else:
            return np.array([self.state_players[s] for s in stateindex])


    # 各行動の人数のリストから、その人数の組み合わせに当たるstateの遷移行列の列番号（行番号）を返す
    def from_stateplayersnum_to_stateindex(self, stateplayersnum):
        stateplayersnum = np.asarray(stateplayersnum)

        # 1次元配列の場合
        if stateplayersnum.ndim == 1:
            if len(stateplayersnum) != self.num_actions:
                print("リストの要素数が不適切です")
                exit(-1)

            if sum(stateplayersnum) != self.num_players:
                print("リストの各要素の和がプレイヤーの総人数と一致しません")
                exit(-1)

            for index, s in enumerate(self.state_players):
                if np.array_equal(s, stateplayersnum):
                    return index

            print("見つかりませんでした")
            exit(-1)

        # 2次元配列の場合、それぞれの要素（1次元配列）を人数の組み合わせと考える
        elif stateplayersnum.ndim == 2:

            for elem in stateplayersnum:
                if len(elem) != self.num_actions:
                    print("リストの要素数が不適切です")
                    exit(-1)

                if sum(elem) != self.num_players:
                    print("リストの各要素の和がプレイヤーの総人数と一致しません")
                    exit(-1)

                for index, s in enumerate(self.state_players):
                    if np.array_equal(s, elem):
                        rst.append(index)

            return rst


    def simulate(self, ts_length, init=None, num_reps=None, **kwargs):
        """
        Simulate the dynamics.

        Parameters
        ----------
        ts_length : scalar(int)
            Time series length of each simulation.

        init : scalar(int) or array_like(int, ndim=1), optional(default=None)
            Number of initial state(s). If None, the initial state is randomly drawn.

        num_reps : scalar(int), optional(default=None)
            Number of repititions of simulation. Relevant only when init is a scalar or None.

        Returns
        -------
        X : ndarray(int, ndim=1 or 2)
            Array containing the sample path(s).
            If of shape (ts_length), if init is a scalar (integer) or None and num_reps is None;
            of shape (k, ts_length) otherwise, where k = len(init) if init is an array_like, otherwise k = num_reps.
        """
        start_init = kwargs.get('start_init', False)

        if init is None:
            if num_reps is None:
                init = np.random.randint(0, self.num_states, 1)
            else:
                init = np.random.randint(0, self.num_states, num_reps)

        elif isinstance(init, int):
            if isinstance(num_reps, int):
                init = [init] * num_reps

        if start_init:
            return np.r_[init, self.mc.simulate(init, ts_length)]
        
        return self.mc.simulate(init, ts_length)


    def plot_simulation(self, ts_length, init=None, num_reps=None, **kwargs):
        if (not (isinstance(init, int) or init is None) or (not (num_reps is None or num_reps == 1))):
            print("複数の試行をplotすることはできません")
            exit(-1)

        simulated = self.simulate(ts_length, init, num_reps, start_init=True)
        if self.num_actions == 2:
            f_list = simulated

        elif self.num_actions > 2:
            # 状態番号のリストから、その状態に対応する人数リストのリストに変換
            simulated = self.from_stateindex_to_stateplayersnum(simulated)
        
        else:
            print("Error!")
            exit(-1)

        fig, ax = plt.subplots()
        plt.title("Simulation of KMR\nInitial state: {0}".format(simulated[0]))
        xlim = kwargs.get('xlim', False)
        ylim = kwargs.get('ylim', False)
        t_list = np.arange(0, ts_length+1, dtype=int)
        
        if self.num_actions == 2:
            plt.plot(t_list, f_list, color='b', linewidth=1, label="Num of action1")
            plt.plot(0, f_list[0], marker='o', color='r', label="Initial point")

        # 行動パターンが3つ以上の場合は、それぞれの行動をとっている人数の推移を折れ線で表現
        if self.num_actions > 2:
            for i in range(self.num_actions):
                f_list = simulated[:, i]
                plt.plot(t_list, f_list, color=cm.gist_rainbow(i*1.0/self.num_actions), linewidth=1, label="action{0}".format(i+1))

            plt.plot(np.zeros(self.num_actions, dtype=int), simulated[0], marker='o', color='r', label="Initial points")

        """
        points_x = [point[0] for point in points]
        points_y = [point[1] for point in points]
        plt.plot(points_x, points_y, 'o', color='r', label="点列")
        """
        
        plt.xlabel("time series")
        plt.ylabel("number of people")
        ax.set_xlim(0, ts_length+1)
        ax.set_ylim(0, self.num_players)

        plt.legend()
        plt.show()


    def compute_stationary_distribution(self):
        """
        Compute the stationary distribution

        Returns
        -------
        stationary_distributions : ndarray(float, ndim=2)
            Array of stationary distribution(s).
        """
        return self.mc.stationary_distributions


if __name__ == '__main__':
    """
    array = [[[4, 4], [0, 3]],
             [[3, 0], [2, 2]]]

    array = [[[6, 6], [0, 5], [0, 0]],
             [[5, 0], [7, 7], [5, 5]],
             [[0, 0], [5, 5], [8, 8]]]

    array = [[4, 0], [3, 2]]
    
    array = [[6, 0, 0],
             [5, 7, 5],
             [0, 5, 8]]
    
    array = [[6, 0, 0],
             [5, 7, 5],
             [0, 5, 8]]
    """

    array = [[6, 0, 0],
             [5, 7, 5],
             [0, 5, 8]]

    kmr = KMR(array, 10, 0.01)
    

    state_players = kmr.state_players
    stationary_distribution = kmr.compute_stationary_distribution()[0]
    for p, d in zip(state_players, stationary_distribution):
        print("{0}: {1:.3f}".format(p, d))


    kmr.plot_simulation(1000)
    


