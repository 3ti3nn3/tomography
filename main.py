import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import mixed
import pure
import const
import general
import visualization
import check
import inversion
import mle
import speed
import simulate


def main(argv = None):

    # 1.1: pure, linear distributed, qutip
    # phi, theta = pure.generate_linear(50, 50)
    # visualization.qubit(states=pure.angles_to_states(phi, theta))

    # 1.2: pure, uniformly distributed, qutip
    # phi, theta = pure.generate_uniform(1000)
    # visualization.qubit(states=pure.angles_to_states(phi, theta))

    # 1.3: pure, uniformly distributed, rotation matrix
    # phi, theta = pure.generate_uniform(1000)
    # rho        = pure.angles_to_density(phi, theta)
    # visualization.qubit(points=np.array([general.expect(sx, rho), general.expect(sy, rho), general.expect(sz, rho)]))

    # 1.3: correcctness check of approach two and three
    # phi, theta = pure.generate_uniform(30)
    # rho        = pure.angles_to_density(phi, theta)
    # visualization.qubit(points=np.array([general.expect(const.sx, rho), general.expect(const.sy, rho), general.expect(const.sz, rho)]), states=pure.angles_to_states(phi, theta), kind='vector', angles=[-60, 30])

    # phi, theta = pure.generate_uniform(100000)
    # rho        = pure.angles_to_density(phi, theta)
    # print('pure: ', check.purity(rho))
    # visualization.expectation_distribution(rho, n_bins=50)

    # 1.4: pure, uniformly distributed, unitary matrices
    # rho = pure.unitary_to_density(2, 1000)
    # visualization.qubit(points=np.array([general.expect(const.sx, rho), general.expect(const.sy, rho), general.expect(const.sz, rho)]))

    # 2.1: mixed, uniformly distributed, bloch vector
    # r, phi, theta = mixed.generate_uniform(10000)
    # rho           = mixed.blochvector_to_density(r, phi, theta)
    # visualization.qubit(points=np.array([general.expect(const.sx, rho), general.expect(const.sy, rho), general.expect(const.sz, rho)]))

    # 2.2: mixed, uniformly distributed, tace method
    # rho = mixed.hermitian_to_density(2, 1000)
    # visualization.qubit(points=np.array([general.expect(const.sx, rho), general.expect(const.sy, rho), general.expect(const.sz, rho)]))
    # visualization.expectation_distribution(rho, n_bins=100)

    # 3.1: speed analysis for pure states
    # pure1 = lambda x: pure.angles_to_states(*pure.generate_uniform(x))
    # pure2 = lambda x: pure.angles_to_density(*pure.generate_uniform(x))
    # pure3 = lambda x: pure.unitary_to_density(2, x)
    # pure4 = lambda x: pure.direct_to_density(*pure.generate_uniform(x))
    #
    # N          = 10000
    # data_pure  = speed.compare(pure1=(pure1, [N]), pure2=(pure2, [N]), pure3=(pure3, [N]), pure4=(pure4, [N]))
    #
    # df = pd.DataFrame.from_dict(data_pure, orient='index')
    # ax = df.plot.bar(figsize=(10, 6), ylabel='time', title='Running time for creating pure states' , legend=False, rot=0)
    # ax.plot()
    # plt.show()

    # 3.2: speed analysis for mixed states
    # mixed1 = lambda x: mixed.blochvector_to_density(*mixed.generate_uniform(x))
    # mixed2 = lambda x: mixed.hermitian_to_density(2, x)
    #
    # N          = 100000
    # data_mixed = speed.compare(mixed1=(mixed1, [N]), mixed2=(mixed2, [N]))
    #
    # df = pd.DataFrame.from_dict(data_mixed, orient='index')
    # ax = df.plot.bar(figsize=(10, 6), ylabel='time', title='Running time for creating mixed states', legend=False, rot=0)
    # ax.plot()
    # plt.show()

    # 4: linear inversion
    # M = np.array([const.se, const.sx, const.sy, const.sz])
    # D = np.array([np.concatenate((np.ones(10)*0, np.ones(10)*1, np.ones(10)*2, np.ones(10)*3)), np.ones(40)]).T
    # n = inversion.count(D, np.zeros((4, 2)))
    #
    # print(inversion.linear(n, M))

    # 5: maximum likelihood estimate
    # d1 = [1, 0]
    # d2 = [2, 0]
    # d3 = [1, 0]
    # a1 = [1, 1]
    # a2 = [2, 1]
    # a3 = [1, 1]
    # D  = np.array([d1, d2, d3])
    # # D  = np.array([d1, d2, d3, a1, a2, a3])
    # print(mle.iterative(D, 1000))

    # 6: simualte quantum measurement
    N     = 1000000

    rho_0 = pure.unitary_to_density(2, 1)
    print(rho_0)

    axes  = np.random.randint(0, high=4, size=N)
    D     = simulate.measure(rho_0, axes)
    print('Measurement simulation finished.')

    rho_1 = simulate.recons(D, M=const.spovm, iter=1000)
    print(rho_1)
    print(general.hilbert_schmidt_distance(rho_0, rho_1))

    rho_2 = simulate.recons(D, M=const.spovm, method='inversion')
    print(rho_2)
    print(general.hilbert_schmidt_distance(rho_0, rho_2))


if __name__ == '__main__':
    main()
