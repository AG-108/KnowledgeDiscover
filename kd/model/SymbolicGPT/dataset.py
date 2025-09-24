#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import multiprocessing as mp
from datetime import datetime
from generator.treeBased.generateData import dataGen
from utils import *  # TODO: replace with a safer import


def processDataTest(numSamples, nv, decimals,
                template, dataPath, fileID, time,
                supportPoints=None,
                supportPointsTest=None,
                numberofPoints=[20, 250],
                xRange=[0.1, 3.1], testPoints=False,
                testRange=[0.0, 6.0], n_levels=3,
                allow_constants=True,
                const_range=[-0.4, 0.4],
                const_ratio=0.8,
                op_list=[
                    "id", "add", "mul", "div",
                    "sqrt", "sin", "exp", "log"],
                sortY=False,
                exponents=[3, 4, 5, 6],
                threshold=1000,
                templatesEQs=None,
                templateProb=0.4,
                ):
    for i in tqdm(range(numSamples)):
        structure = template.copy()
        # generate a formula
        # Create a new random equation
        try:
            _, skeletonEqn, _ = dataGen(
                nv=nv, decimals=decimals,
                numberofPoints=numberofPoints,
                supportPoints=supportPoints,
                supportPointsTest=supportPointsTest,
                xRange=xRange,
                testPoints=testPoints,
                testRange=testRange,
                n_levels=n_levels,
                op_list=op_list,
                allow_constants=allow_constants,
                const_range=const_range,
                const_ratio=const_ratio,
                exponents=exponents
            )
            if templatesEQs != None and np.random.rand() < templateProb:
                # by a chance, replace the skeletonEqn with a given templates
                idx = np.random.randint(len(templatesEQs[nv]))
                skeletonEqn = templatesEQs[nv][idx]

        except Exception as e:
            # Handle any exceptions that timing might raise here
            print("\n-->dataGen(.) was terminated!\n{}\n".format(e))
            i = i - 1
            continue

        # fix exponents that are larger than our expected value, sometimes the data generator generates those odd numbers
        exps = re.findall(r"(\*\*[0-9\.]+)", skeletonEqn)
        for ex in exps:
            # correct the exponent
            cexp = '**' + str(eval(ex[2:]) if eval(ex[2:]) < exponents[-1] else np.random.randint(2, exponents[-1] + 1))
            # replace the exponent
            skeletonEqn = skeletonEqn.replace(ex, cexp)

            # replace the constants with new ones
        cleanEqn = ''
        for chr in skeletonEqn:
            if chr == 'C':
                # genereate a new random number
                chr = '{}'.format(np.random.uniform(const_range[0], const_range[1]))
            cleanEqn += chr

        if 'I' in cleanEqn or 'zoo' in cleanEqn:
            # repeat the equation generation
            print('This equation has been rejected: {}'.format(cleanEqn))
            i -= 1
            continue

        # create a set of points
        nPoints = np.random.randint(
            *numberofPoints) if supportPoints is None else len(supportPoints)

        data = generateDataStrEq(cleanEqn, n_points=nPoints, n_vars=nv,
                                 decimals=decimals, supportPoints=supportPoints, min_x=xRange[0], max_x=xRange[1])
        # use the new x and y
        x, y = data

        if testPoints:
            dataTest = generateDataStrEq(cleanEqn, n_points=nPoints, n_vars=nv, decimals=decimals,
                                         supportPoints=supportPointsTest, min_x=testRange[0], max_x=testRange[1])
            xT, yT = dataTest

        # check if there is nan/inf/very large numbers in the y
        if np.isnan(y).any() or np.isinf(y).any() or np.any([abs(e) > threshold for e in y]):
            # repeat the equation generation
            i -= 1
            print('{} has been rejected because of wrong value in y.'.format(skeletonEqn))
            continue

        if len(y) == 0:  # if for whatever reason the y is empty
            print('Empty y, x: {}, most of the time this is because of wrong number_of_points: {}'.format(x,
                                                                                                        numberofPoints))
            e -= 1
            continue

        # just make sure there is no samples out of the threshold
        if abs(min(y)) > threshold or abs(max(y)) > threshold:
            raise 'Err: Min:{},Max:{},Threshold:{}, \n Y:{} \n Eq:{}'.format(min(y), max(y), threshold, y, cleanEqn)

        # sort data based on Y
        if sortY:
            x, y = zip(*sorted(zip(x, y), key=lambda d: d[1]))

        # hold data in the structure
        structure['X'] = list(x)
        structure['Y'] = y
        structure['EQ'] = cleanEqn
        structure['Skeleton'] = skeletonEqn
        structure['XT'] = list(xT)
        structure['YT'] = yT

        print('\n EQ: {}'.format(skeletonEqn))

        outputPath = dataPath.format(fileID, nv, time)
        if os.path.exists(outputPath):
            fileSize = os.path.getsize(outputPath)
            if fileSize > 500000000:  # 500 MB
                fileID += 1
        with open(outputPath, "a", encoding="utf-8") as h:
            json.dump(structure, h, ensure_ascii=False)
            h.write('\n')

def processData(num_samples, nv, decimals,
                template, data_path, fileID, time,
                support_points=None,
                support_points_test=None,
                number_of_points=None,
                xRange=[0.1, 3.1], testPoints=False,
                testRange=[0.0, 6.0], n_levels=3,
                allow_constants=True,
                const_range=[-0.4, 0.4],
                const_ratio=0.8,
                op_list=[
                    "id", "add", "mul", "div",
                    "sqrt", "sin", "exp", "log"],
                sortY=False,
                exponents=[3, 4, 5, 6],
                numSamplesEachEq=1,
                threshold=100,
                templatesEQs=None,
                templateProb=0.4,
                ):
    for i in tqdm(range(num_samples)):
        structure = template.copy()
        # generate a formula
        # Create a new random equation
        try:
            _, skeletonEqn, _ = dataGen(
                nv=nv, decimals=decimals,
                numberofPoints=number_of_points,
                supportPoints=support_points,
                supportPointsTest=support_points_test,
                xRange=xRange,
                testPoints=testPoints,
                testRange=testRange,
                n_levels=n_levels,
                op_list=op_list,
                allow_constants=allow_constants,
                const_range=const_range,
                const_ratio=const_ratio,
                exponents=exponents
            )
            if templatesEQs != None and np.random.rand() < templateProb:
                # by a chance, replace the skeletonEqn with a given templates
                idx = np.random.randint(len(templatesEQs[nv]))
                skeletonEqn = templatesEQs[nv][idx]

        except Exception as e:
            # Handle any exceptions that timing might raise here
            print("\n-->dataGen(.) was terminated!\n{}\n".format(e))
            i = i - 1
            continue

        # fix exponents that are larger than our expected value, sometimes the data generator generates those odd numbers
        exps = re.findall(r"(\*\*[0-9\.]+)", skeletonEqn)
        for ex in exps:
            # correct the exponent
            cexp = '**' + str(eval(ex[2:]) if eval(ex[2:]) < exponents[-1] else np.random.randint(2, exponents[-1] + 1))
            # replace the exponent
            skeletonEqn = skeletonEqn.replace(ex, cexp)

        for e in range(numSamplesEachEq):
            # replace the constants with new ones
            cleanEqn = ''
            for chr in skeletonEqn:
                if chr == 'C':
                    # genereate a new random number
                    chr = '{}'.format(np.random.uniform(const_range[0], const_range[1]))
                cleanEqn += chr

            if 'I' in cleanEqn or 'zoo' in cleanEqn:
                # repeat the equation generation
                print('This equation has been rejected: {}'.format(cleanEqn))
                i -= 1  # TODO: this might lead to a bad loop
                break

            # generate new data points
            nPoints = np.random.randint(
                *number_of_points) if support_points is None else len(support_points)

            try:
                data = generateDataStrEq(cleanEqn, n_points=nPoints, n_vars=nv,
                                         decimals=decimals, supportPoints=support_points, min_x=xRange[0],
                                         max_x=xRange[1])
            except:
                # for different reason this might happend including but not limited to division by zero
                continue
            # if testPoints:
            #     dataTest = generateDataStrEq(currEqn, n_points=number_of_points, n_vars=nv, decimals=decimals,
            #                                  support_points=support_points_test, min_x=testRange[0], max_x=testRange[1]))

            # use the new x and y
            x, y = data

            # check if there is nan/inf/very large numbers in the y
            if np.isnan(y).any() or np.isinf(y).any():  # TODO: Later find a more optimized solution
                # repeat the data generation
                # i -= 1 #TODO: this might lead to a bad loop
                # break
                e -= 1
                continue

            # replace out of threshold with maximum numbers
            y = [e if abs(e) < threshold else np.sign(e) * threshold for e in y]

            if len(y) == 0:  # if for whatever reason the y is empty
                print('Empty y, x: {}, most of the time this is because of wrong number_of_points: {}'.format(x,
                                                                                                            number_of_points))
                e -= 1
                continue

            # just make sure there is no samples out of the threshold
            if abs(min(y)) > threshold or abs(max(y)) > threshold:
                raise 'Err: Min:{},Max:{},Threshold:{}, \n Y:{} \n Eq:{}'.format(min(y), max(y), threshold, y, cleanEqn)

            # sort data based on Y
            if sortY:
                x, y = zip(*sorted(zip(x, y), key=lambda d: d[1]))

            # hold data in the structure
            structure['X'] = list(x)
            structure['Y'] = y
            structure['Skeleton'] = skeletonEqn
            structure['EQ'] = cleanEqn

            outputPath = data_path.format(fileID, nv, time)
            if os.path.exists(outputPath):
                fileSize = os.path.getsize(outputPath)
                if fileSize > 500000000:  # 500 MB
                    fileID += 1
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
    args = parser.parse_args()

    # setting working directory
    data_folder = os.path.join(args.workdir, 'Dataset')
    config_path = os.path.join(data_folder, f'{args.type}_cfg.json')

    # load the config file
    cfg = json.load(open(config_path, 'r'))

    # setting up the parameters
    seed = cfg.get('seed', 2021)
    # 2021 for Train, 2022 for Val, 2023 for Test, you have to change the generateData.py seed as well
    num_vars = cfg.get('num_vars', list(range(1, 4)))
    # NOTE: For linux you can only use unique num_vars, in Windows, it is possible to use [1,2,3,4] * 10!
    decimals = cfg.get('decimals', 4)
    number_of_points = cfg.get('number_of_points', [20, 250])  # only usable if support points has not been provided
    num_samples = cfg.get('num_samples', 100)  # number of generated samples
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
    data_path = os.path.join(data_folder, '{}_{}_{}.json')

    print(os.mkdir(data_folder) if not os.path.isdir(data_folder) else 'We do have the path already!')

    generating_template = {'X': [], 'Y': 0.0, 'EQ': ''}
    fileID = 0
    # mp.set_start_method('spawn')
    # q = mp.Queue()
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
                               fileID,
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
                               fileID,
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
