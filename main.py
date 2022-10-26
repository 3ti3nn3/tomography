import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import qutip as qt

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

    # 4: simulate quantum measurement and reconstruct according to linear inversion
    # N     = int(1e06)
    # rho_0 = pure.unitary_to_density(2, 1)
    # M     = const.pauli4
    #
    # D       = simulate.measure(rho_0, N, M)
    # rho_inv = inversion.linear(D, M)
    #
    # visualization.qubit_3(('original', rho_0), ('inv', [rho_inv]))

    # 5: simulate quantum measurement and reconstruct according to maximum likelihood estimate
    # N     = int(1e04)
    # rho_0 = pure.unitary_to_density(2, 1)
    # M     = const.pauli6
    #
    # D       = simulate.measure(rho_0, N, M)
    # rho_mle = mle.iterative(D, M, 100)
    #
    # visualization.qubit_3(('original', rho_0), ('mle', [rho_mle]))

    # 6: check convergence of maximum likelihood estimate
    # N     = int(1e03)
    # rho_0 = pure.unitary_to_density(2, 1)
    # M     = const.pauli4
    #
    # visualization.hilbert_dist(rho_0, M, N)

    # 7: check iteration dependency of Hilbert-Schmidt distance
    # iter_max = 100
    # rho_0    = pure.unitary_to_density(2, 1)
    # M        = const.pauli4
    #
    # visualization.hilbert_dist_iter(rho_0, M, iter_max)

    # 8: plot bures distance
    # N     = np.int64(1e06)
    # rho_0 = pure.unitary_to_density(2, 1)
    # M     = const.pauli4
    #
    # visualization.bures_dist(rho_0, M, N)

    # 9: plot infidelity
    N     = np.int64(1e05)

    # eigenstates
    # rho_0 = 1/2*np.array([[1, 1], [1, 1]])
    # rho_0 = 1/2*np.array([[1, -1j], [1j, 1]])
    # rho_0 = np.array([[1, 0], [0, 0]])
    # rho_0 = np.array([-0.32505758+0.32505758j, 0.88807383+0.j])[:, None]@np.array([-0.32505758-0.32505758j, 0.88807383+0.j])[None, :]

    # non eigenstates
    # rho_0 = 1/2*np.array([[1, -1], [-1, 1]])
    # rho_0 = 1/2*np.array([[1, 1j], [-1j, 1]])
    # rho_0 = np.array([[0, 0], [0, 1]])

    # other
    rho_0 = mixed.hermitian_to_density(2, 1)[0]
    # rho_0 = pure.unitary_to_density(2, 1)

    # POVMs
    M1 = const.pauli4
    M2 = const.pauli6

    func1 = ('MLE with Pauli4', mle.iterative)
    func2 = ('MLE with Pauli6', mle.iterative)

    visualization.infidelity(rho_0, func1, func2, M1, M2, N)


if __name__ == '__main__':
    main()
