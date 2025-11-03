#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import multiprocessing as mp
from datetime import datetime
from typing import List, Tuple, Dict

from generator.treeBased.generateData import dataGen
from utils import *  # TODO: replace with a safer import


def processDataTest(num_samples, nv, decimals,
                template, data_path, file_id, time,
                support_points: np.array=None,
                support_points_test: np.array=None,
                number_of_points: List[int]=None,
                x_range: List[int]=None, test_points=False,
                test_range: List[int]=None, n_levels=3,
                allow_constants=True,
                const_range: List[int]=None,
                const_ratio=0.8,
                op_list: List[str]=None,
                sortY=False,
                exponents: List[int]=None,
                threshold=1000,
                template_eqs: Dict[int, str]=None,
                template_prob=0.4,
                ):
    for i in tqdm(range(num_samples)):
        structure = template.copy()
        # generate a formula
        # Create a new random equation
        try:
            _, skeleton_eqn, _ = dataGen(
                nv=nv, decimals=decimals,
                numberofPoints=number_of_points,
                supportPoints=support_points,
                supportPointsTest=support_points_test,
                xRange=x_range,
                testPoints=test_points,
                testRange=test_range,
                n_levels=n_levels,
                op_list=op_list,
                allow_constants=allow_constants,
                const_range=const_range,
                const_ratio=const_ratio,
                exponents=exponents
            )
            if template_eqs != None and np.random.rand() < template_prob:
                # by a chance, replace the skeleton_eqn with a given templates
                idx = np.random.randint(len(template_eqs[nv]))
                skeleton_eqn = template_eqs[nv][idx]

        except Exception as e:
            # Handle any exceptions that timing might raise here
            print("\n-->dataGen(.) was terminated!\n{}\n".format(e))
            # 这行代码完全没用，并不会让循环重来；但是暂时不要改动
            i = i - 1
            continue

        # fix exponents that are larger than our expected value, sometimes the data generator generates those odd numbers
        exps = re.findall(r"(\*\*[0-9\.]+)", skeleton_eqn)
        for ex in exps:
            # correct the exponent
            cexp = '**' + str(eval(ex[2:]) if eval(ex[2:]) < exponents[-1] else np.random.randint(2, exponents[-1] + 1))
            # replace the exponent
            skeleton_eqn = skeleton_eqn.replace(ex, cexp)

            # replace the constants with new ones
        clean_eqn = ''
        for chr in skeleton_eqn:
            if chr == 'C':
                # genereate a new random number
                chr = '{}'.format(np.random.uniform(const_range[0], const_range[1]))
            clean_eqn += chr

        if 'I' in clean_eqn or 'zoo' in clean_eqn:
            # repeat the equation generation
            print('This equation has been rejected: {}'.format(clean_eqn))
            i -= 1
            continue

        # create a set of points
        nPoints = np.random.randint(
            *number_of_points) if support_points is None else len(support_points)

        data = generateDataStrEq(clean_eqn, n_points=nPoints, n_vars=nv,
                                 decimals=decimals, supportPoints=support_points, min_x=x_range[0], max_x=x_range[1])
        # use the new x and y
        x, y = data

        if test_points:
            dataTest = generateDataStrEq(clean_eqn, n_points=nPoints, n_vars=nv, decimals=decimals,
                                         supportPoints=support_points_test, min_x=test_range[0], max_x=test_range[1])
            xT, yT = dataTest

        # check if there is nan/inf/very large numbers in the y
        if np.isnan(y).any() or np.isinf(y).any() or np.any([abs(e) > threshold for e in y]):
            # repeat the equation generation
            i -= 1
            print('{} has been rejected because of wrong value in y.'.format(skeleton_eqn))
            continue

        if len(y) == 0:  # if for whatever reason the y is empty
            print('Empty y, x: {}, most of the time this is because of wrong number_of_points: {}'.format(x,
                                                                                                          number_of_points))
            continue

        # just make sure there is no samples out of the threshold
        if abs(min(y)) > threshold or abs(max(y)) > threshold:
            raise 'Err: Min:{},Max:{},Threshold:{}, \n Y:{} \n Eq:{}'.format(min(y), max(y), threshold, y, clean_eqn)

        # sort data based on Y
        if sortY:
            x, y = zip(*sorted(zip(x, y), key=lambda d: d[1]))

        # hold data in the structure
        structure['X'] = list(x)
        structure['Y'] = y
        structure['EQ'] = clean_eqn
        structure['Skeleton'] = skeleton_eqn
        structure['XT'] = list(xT)
        structure['YT'] = yT

        print('\n EQ: {}'.format(skeleton_eqn))

        outputPath = data_path.format(file_id, nv, time)
        if os.path.exists(outputPath):
            fileSize = os.path.getsize(outputPath)
            if fileSize > 500000000:  # 500 MB
                file_id += 1
        with open(outputPath, "a", encoding="utf-8") as h:
            json.dump(structure, h, ensure_ascii=False)
            h.write('\n')

def processData(num_samples, nv, decimals,
                template, data_path, file_id, time,
                support_points=None,
                support_points_test=None,
                number_of_points=None,
                x_range: List[int]=None, test_points=False,
                test_range: List[int]=None, n_levels=3,
                allow_constants=True,
                const_range: List[int]=None,
                const_ratio=0.8,
                op_list: List[str]=None,
                sortY=False,
                exponents: List[int]=None,
                num_samples_each_eq=1,
                threshold=100,
                templates_eqs=None,
                template_prob=0.4,
                ):
    for i in tqdm(range(num_samples)):
        structure = template.copy()
        # generate a formula
        # Create a new random equation
        try:
            _, skeleton_eqn, _ = dataGen(
                nv=nv, decimals=decimals,
                numberofPoints=number_of_points,
                supportPoints=support_points,
                supportPointsTest=support_points_test,
                xRange=x_range,
                testPoints=test_points,
                testRange=test_range,
                n_levels=n_levels,
                op_list=op_list,
                allow_constants=allow_constants,
                const_range=const_range,
                const_ratio=const_ratio,
                exponents=exponents
            )
            if templates_eqs is not None and np.random.rand() < template_prob:
                # by a chance, replace the skeleton_eqn with a given templates
                idx = np.random.randint(len(templates_eqs[nv]))
                skeleton_eqn = templates_eqs[nv][idx]

        except Exception as e:
            # Handle any exceptions that timing might raise here
            print("\n-->dataGen(.) was terminated!\n{}\n".format(e))
            i = i - 1
            continue

        # fix exponents that are larger than our expected value, sometimes the data generator generates those odd numbers
        exps = re.findall(r"(\*\*[0-9\.]+)", skeleton_eqn)
        for ex in exps:
            # correct the exponent
            cexp = '**' + str(eval(ex[2:]) if eval(ex[2:]) < exponents[-1] else np.random.randint(2, exponents[-1] + 1))
            # replace the exponent
            skeleton_eqn = skeleton_eqn.replace(ex, cexp)

        for e in range(num_samples_each_eq):
            # replace the constants with new ones
            clean_eqn = ''
            for chr in skeleton_eqn:
                if chr == 'C':
                    # genereate a new random number
                    chr = '{}'.format(np.random.uniform(const_range[0], const_range[1]))
                clean_eqn += chr

            if 'I' in clean_eqn or 'zoo' in clean_eqn:
                # repeat the equation generation
                print('This equation has been rejected: {}'.format(clean_eqn))
                i -= 1  # 原作者以为这里会有死循环，其实没有；但是是否要改成他想象中的效果有待商榷
                break

            # generate new data points
            nPoints = np.random.randint(
                *number_of_points) if support_points is None else len(support_points)

            try:
                data = generateDataStrEq(clean_eqn, n_points=nPoints, n_vars=nv,
                                         decimals=decimals, supportPoints=support_points, min_x=x_range[0],
                                         max_x=x_range[1])
            except:
                # for different reason this might happend including but not limited to division by zero
                continue
            # if test_points:
            #     dataTest = generateDataStrEq(currEqn, n_points=number_of_points, n_vars=nv, decimals=decimals,
            #                                  support_points=support_points_test, min_x=test_range[0], max_x=test_range[1]))

            # use the new x and y
            x, y = data

            # check if there is nan/inf/very large numbers in the y
            if np.isnan(y).any() or np.isinf(y).any():  # TODO: Later find a more optimized solution
                continue

            # replace out of threshold with maximum numbers
            y = [e if abs(e) < threshold else np.sign(e) * threshold for e in y]

            if len(y) == 0:  # if for whatever reason the y is empty
                print('Empty y, x: {}, most of the time this is because of wrong number_of_points: {}'.format(x,
                                                                                                            number_of_points))
                continue

            # just make sure there is no samples out of the threshold
            if abs(min(y)) > threshold or abs(max(y)) > threshold:
                raise 'Err: Min:{},Max:{},Threshold:{}, \n Y:{} \n Eq:{}'.format(min(y), max(y), threshold, y, clean_eqn)

            # sort data based on Y
            if sortY:
                x, y = zip(*sorted(zip(x, y), key=lambda d: d[1]))

            # hold data in the structure
            structure['X'] = list(x)
            structure['Y'] = y
            structure['Skeleton'] = skeleton_eqn
            structure['EQ'] = clean_eqn

            outputPath = data_path.format(file_id, nv, time)
            if os.path.exists(outputPath):
                fileSize = os.path.getsize(outputPath)
                if fileSize > 500000000:  # 500 MB
                    file_id += 1
                    # 这里更新了file_id却没有更新outputPath
            with open(outputPath, "a", encoding="utf-8") as h:
                json.dump(structure, h, ensure_ascii=False)
                h.write('\n')


def main():
    # Config

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--type', type=str, default='train',
                        help='Choose the type of dataset to generate [train/val/test]')
    parser.add_argument('--workdir', type=str, default='.', help='The working directory')
    parser.add_argument('--datadir', type=str, default='datasets/NewExperiments', help='The data directory')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed')
    args = parser.parse_args()

    # setting working directory
    config_dir = os.path.join(args.workdir, 'Dataset')
    config_path = os.path.join(config_dir, f'{args.type}_cfg.json')

    # load the config file
    cfg = json.load(open(config_path, 'r'))

    # setting up the parameters
    seed = args.seed
    # 2021 for Train, 2022 for Val, 2023 for Test, you have to change the generateData.py seed as well
    num_vars = cfg.get('num_vars', list(range(1, 4)))
    # NOTE: For linux you can only use unique num_vars, in Windows, it is possible to use [1,2,3,4] * 10!
    decimals = cfg.get('decimals', 4)
    number_of_points = cfg.get('number_of_points', [20, 250])  # only usable if support points has not been provided
    num_samples = cfg.get('num_samples', 100)  # number of generated samples
    if args.type == 'test' or args.type == 'val':
        num_samples = 1000 // len(num_vars)
    test_points = cfg.get('test_points', False)
    train_range = cfg.get('train_range', [-3.0, 3.0])
    test_range = cfg.get('test_range', [[-5.0, 3.0], [-3.0, 5.0]])  # this means Union((-5,-1),(1,5))
    support_points = cfg.get('support_points', None)
    support_points_test = cfg.get('support_points_test', None)
    n_levels = cfg.get('n_levels', 4)
    allow_constants = cfg.get('allow_constants', True)
    const_range = cfg.get('const_range', [-2.1, 2.1])
    const_ratio = cfg.get('const_ratio', 0.5)
    op_list = cfg.get('op_list', [
        "id", "add", "mul",
        "sin", "pow", "cos", "sqrt",
        "exp", "div", "sub", "log",
        "arcsin",
    ])
    exponents = cfg.get('exponents', [3, 4, 5, 6])
    sortY = cfg.get('sortY', False)  # if the data is sorted based on y
    num_samples_each_eq = cfg.get('num_samples_each_eq', 5)
    threshold = cfg.get('threshold', 5000)
    template_prob = cfg.get('template_prob', 0.1)  # the probability of generating an equation from the templates
    template_eqs = cfg.get('template_eqs',
                           None)  # template equations, if NONE then there will be no specific templates for the generated equations
    # transform the keys to int due to the restriction of json
    temp = {int(k): v for k, v in template_eqs.items()} if template_eqs is not None else None
    template_eqs = temp

    # setting up random seed
    import random
    random.seed(seed)
    np.random.seed(seed=seed)  # fix the seed for reproducibility

    # the output file name template
    output_dir = os.path.join(args.workdir, args.datadir, args.type.capitalize())
    data_path = os.path.join(output_dir, '{}_{}_{}.json')

    print(os.mkdir(output_dir) if not os.path.isdir(output_dir) else 'We do have the path already!')

    generating_template = {'X': [], 'Y': 0.0, 'EQ': ''}
    file_id = 0
    processes = []
    for i, nv in enumerate(num_vars):
        now = datetime.now()
        time = '{}_'.format(i) + now.strftime("%d%m%Y_%H%M%S")
        print('Processing equations with {} variables!'.format(nv))
        if args.type == 'val' or args.type == 'test':
            p = mp.Process(target=processDataTest,
                           args=(
                               num_samples,
                               nv,
                               decimals,
                               generating_template,
                               data_path,
                               file_id,
                               time,
                               support_points,
                               support_points_test,
                               number_of_points,
                               train_range,
                               test_points,
                               test_range,
                               n_levels,
                               allow_constants,
                               const_range,
                               const_ratio,
                               op_list,
                               sortY,
                               exponents,
                               threshold,
                               template_eqs,
                               template_prob
                           )
                           )
        else:
            p = mp.Process(target=processData,
                           args=(
                               num_samples,
                               nv,
                               decimals,
                               generating_template,
                               data_path,
                               file_id,
                               time,
                               support_points,
                               support_points_test,
                               number_of_points,
                               train_range,
                               test_points,
                               test_range,
                               n_levels,
                               allow_constants,
                               const_range,
                               const_ratio,
                               op_list,
                               sortY,
                               exponents,
                               num_samples_each_eq,
                               threshold,
                               template_eqs,
                               template_prob
                           )
                           )

        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
