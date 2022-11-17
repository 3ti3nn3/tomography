def generate_linear(n_phi: int, n_theta: int):
    '''
    Generates linear distributed states,i.e. linear distributed angles in the allowed range.

    :param n_phi  : number of linear distributed polar points
    :param n_theta: number of linear distributed azimuth points
    :return: tuple of two arrays with angles
    '''
    phi             = np.linspace(0, 2*np.pi, n_phi)
    theta           = np.linspace(0, np.pi, n_theta)
    phi_v, theta_v  = np.meshgrid(phi, theta)

    return phi_v.flatten(), theta_v.flatten()


def angles_to_states(phi, theta):
    '''
    Takes angles and converts them into states in the compuational basis.

    :param phi  : array or float of polar angles
    :param theta: array or float of azimuth angles
    return: Qobj in computational basis
    '''
    up     = qt.basis(2, 0)
    down   = qt.basis(2, 1)

    return np.cos(theta/2).tolist()*up + ( np.sin(theta/2) * np.exp(1j*phi) ).tolist()*down


def measure_slow(rho: np.array, N: int):
    '''
    Simulates several quantum measurements for a set of operators.

    :param rho: state to sample from
    :param N  : sample size
    :return: array of N measured results and the corresponding axis
    '''
    axs = np.random.randint(0, high=3, size=N)
    p1  = np.trace([3*rho@const.M_up[ax] for ax in axs], axis1=-2, axis2=-1)

    if np.all(np.imag(p1)<1e-14):
        p1 = np.real(p1)
    else:
        raise ValueError('Contradiction: Complex probabilities!')

    p      = np.array([1-p1, p1]).T
    choice = lambda p: np.random.choice([0, 1], p=p)

    return np.array([axs, np.apply_along_axis(choice, 1, p)]).T


def measure_unefficiently(rho: np.array, N: int):
    '''
    Simulates several quantum measurements for a set of operators.

    :param rho: state to sample from
    :param N  : sample size
    :return: array of N measured results and the corresponding axis
    '''
    axs = np.random.randint(0, high=3, size=N)
    ax, ax_counts = np.unique(axs, return_counts=True)

    p10 = np.real(np.trace(3*rho@const.M_up[ax[0]]))
    p11 = np.real(np.trace(3*rho@const.M_up[ax[1]]))
    p12 = np.real(np.trace(3*rho@const.M_up[ax[2]]))

    D0 = np.array([np.repeat(0, ax_counts[0]), np.random.choice([0, 1], p=[1-p10, p10], size=ax_counts[0])]).T
    D1 = np.array([np.repeat(1, ax_counts[1]), np.random.choice([0, 1], p=[1-p11, p11], size=ax_counts[1])]).T
    D2 = np.array([np.repeat(2, ax_counts[2]), np.random.choice([0, 1], p=[1-p12, p12], size=ax_counts[2])]).T

    D = np.concatenate((D0, D1, D2), axis=0)
    np.random.shuffle(D)

    return D


def distance(func_dist: tuple, func_rho: tuple, func_1: tuple, func_2: tuple, N_max: int, N_rho: int, N_N: int, dim=2):
    '''
    Plots the N dependcy of given distance measure

    :param func_dist: distance measure
        datatype: tuple (description: str, distance measure: function)
    :param func_rho : function which randomly creates states
        datatype: (description: str, sample state function: function)
    :param func_1   : first method how to reconstruct the true state
        datatype: tuple (description: str, reconstruction method: function, POVM: np.array)
    :param func_2   : second method how to reconstruct the true state
        datatype: tuple (description: str, reconstruction method: function, POVM: np.array)
    :param N_max    : maximal number of total measurements
    :param N_rho    : total number of distinct states to average over
    :param N_N      : number of different N
    :param dim      : dimension of the considered system
    :return:
    '''
    x_N = np.logspace(2, np.log10(N_max), N_N, dtype=np.int64)

    # initialize storage
    rho_m1   = np.zeros((N_rho, N_N, dim, dim), dtype=np.complex)
    rho_m2   = np.zeros((N_rho, N_N, dim, dim), dtype=np.complex)
    valid_m1 = np.zeros((N_rho, N_N), dtype=bool)
    valid_m2 = np.zeros((N_rho, N_N), dtype=bool)
    dist_m1  = np.zeros((N_rho, N_N), dtype=np.float)
    dist_m2  = np.zeros((N_rho, N_N), dtype=np.float)

    # define validity function
    if func_rho[1] is pure.unitary_to_density:
        validity = lambda rho: check.state(rho) and check.purity(rho)
    else:
        validity = lambda rho: check.state(rho)

    # fill up arrays
    for j in range(N_rho):
        rho = func_rho[1](dim, 1)

        # create data set
        if np.all(func_1[2]==func_2[2]):
            D  = simulate.measure(rho, N_max, func_1[2])
            D1 = D
            D2 = D
        else:
            D1 = simulate.measure(rho, N_max, func_1[2])
            D2 = simulate.measure(rho, N_max, func_2[2])

        # reconstruct
        for i in range(N_N):
            rho_m1[j,i]   = func_1[1](D1[:x_N[i]], func_1[2])
            dist_m1[j,i]  = func_dist[1](rho, rho_m1[j,i])
            valid_m1[j,i] = validity(rho)

            rho_m2[j,i]   = func_2[1](D2[:x_N[i]], func_2[2])
            dist_m2[j,i]  = func_dist[1](rho, rho_m2[j,i])
            valid_m2[j,i] = validity(rho)


    print('validity for method 1:', np.sum(valid_m1==True)/(N_rho*N_N))
    print('validity for method 2:', np.sum(valid_m2==True)/(N_rho*N_N))

    # calculates mean
    mean_m1 = np.mean(dist_m1, axis=0, where=valid_m1)
    mean_m2 = np.mean(dist_m2, axis=0, where=valid_m2)

    std_m1  = np.std(dist_m1, axis=0, where=valid_m1)
    std_m2  = np.std(dist_m1, axis=0, where=valid_m2)

    # build plots
    plt.figure(figsize=(12, 9))
    plt.title(f'N-scaling of {func_dist[0]} for {func_rho[0]}')

    # fit curve
    try:
        f = lambda x, a, A: A*x**a

        popt_m1, pcov_m1 = curve_fit(f, x_N, mean_m1, p0=[-0.5, 0.2], sigma=std_m1)
        popt_m2, pcov_m2 = curve_fit(f, x_N, mean_m2, p0=[-0.5, 0.2], sigma=std_m2)

        param1_m1 = popt_m1-np.sqrt(np.diag(pcov_m1))
        param2_m1 = popt_m1+np.sqrt(np.diag(pcov_m1))

        param1_m2 = popt_m2-np.sqrt(np.diag(pcov_m2))
        param2_m2 = popt_m2+np.sqrt(np.diag(pcov_m2))

        x = np.logspace(2, np.log10(N_max), 100, dtype=np.int64)
        plt.fill_between(x, y1=f(x, *param1_m1), y2=f(x, *param2_m1), color='lightblue', alpha=0.3, label=f'$1\sigma$ of {func_1[0]}')
        plt.fill_between(x, y1=f(x, *param1_m2), y2=f(x, *param2_m2), color='lightgreen', alpha=0.3, label=f'$1\sigma$ of {func_2[0]}')

        plt.plot(x, f(x, *popt_m1), color='lightblue', label='Fit {0}, a = {1:.2e} $\pm$ {2:.2e}'.format(func_1[0], popt_m1[0], np.sqrt(pcov_m1[0, 0])))
        plt.plot(x, f(x, *popt_m2), color='lightgreen', label='Fit {0}, a = {1:.2e} $\pm$ {2:.2e}'.format(func_2[0], popt_m2[0], np.sqrt(pcov_m2[0, 0])))
    except:
        print('Fitting was not successful!')

    plt.plot(x_N, mean_m1, color='navy', linestyle='None', markersize=5, marker='o', label=func_1[0])
    plt.plot(x_N, mean_m2, color='forestgreen', linestyle='None', markersize=5, marker='o', label=func_2[0])

    plt.xlabel(r'$N$')
    plt.ylabel(f'{func_dist[0]}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(x_N[0], x_N[-1])

    plt.legend()
    plt.show()


def distance_adaptive(func_dist: tuple, func_rho: tuple, func_1: tuple, func_2: tuple, N_max: int, N_rho: int, N_N: int, dim=2):
    '''
    Plots the N dependcy of given distance measure

    :param func_dist: distance measure
        datatype: tuple (description: str, distance measure: function)
    :param func_rho : function which randomly creates states
        datatype: (description: str, sample state function: function)
    :param func_1   : first method how to reconstruct the true state
        datatype: tuple (description: str, reconstruction method: function, POVM: np.array)
    :param func_2   : second method how to reconstruct the true state
        datatype: tuple (description: str, reconstruction method: function, POVM: np.array)
    :param N_max    : maximal number of total measurements
    :param N_rho    : total number of distinct states to average over
    :param N_N      : number of different points
    :param dim      : dimension of the considered system
    :return:
    '''
    x_N = np.logspace(2, np.log10(N_max), N_N, dtype=np.int64)

    # initialize storage
    rho_m1   = np.zeros((N_rho, N_N, dim, dim), dtype=np.complex)
    rho_m2   = np.zeros((N_rho, N_N, dim, dim), dtype=np.complex)
    valid_m1 = np.zeros((N_rho, N_N), dtype=bool)
    valid_m2 = np.zeros((N_rho, N_N), dtype=bool)
    dist_m1  = np.zeros((N_rho, N_N), dtype=np.float)
    dist_m2  = np.zeros((N_rho, N_N), dtype=np.float)

    # define validity function
    if func_rho[1] is pure.unitary_to_density:
        validity = lambda rho: check.state(rho) and check.purity(rho)
    else:
        validity = lambda rho: check.state(rho)

    # fill up arrays
    for j in range(N_rho):
        rho = func_rho[1](dim, 1)

        # reconstruct
        for i in range(N_N):
            rho_m1[j,i]   = func_1[1](rho, func_1[2], x_N[i])
            dist_m1[j,i]  = func_dist[1](rho, rho_m1[j,i])
            valid_m1[j,i] = validity(rho)

            rho_m2[j,i]   = func_2[1](rho, func_2[2], x_N[i])
            dist_m2[j,i]  = func_dist[1](rho, rho_m2[j,i])
            valid_m2[j,i] = validity(rho)

    print('validity for method 1:', np.sum(valid_m1==True)/(N_rho*N_N))
    print('validity for method 2:', np.sum(valid_m2==True)/(N_rho*N_N))

    print('number of negative infidelity for method 1:', np.sum(dist_m1<0)/(N_rho*N_N))
    print('number of negative infidelity for method 2:', np.sum(dist_m2<0)/(N_rho*N_N))

    # calculates mean
    mean_m1 = np.mean(dist_m1, axis=0, where=valid_m1)
    mean_m2 = np.mean(dist_m2, axis=0, where=valid_m2)

    std_m1  = np.std(dist_m1, axis=0, where=valid_m1)
    std_m2  = np.std(dist_m1, axis=0, where=valid_m2)

    # build plots
    plt.figure(figsize=(12, 9))
    plt.title(f'N-scaling of {func_dist[0]} for {func_rho[0]} averaged over {N_rho} states')

    try:
        # fit curve
        f = lambda x, a, A: A*x**a

        popt_m1, pcov_m1 = curve_fit(f, x_N, mean_m1, p0=[-0.5, 0.2], sigma=std_m1)
        popt_m2, pcov_m2 = curve_fit(f, x_N, mean_m2, p0=[-0.5, 0.2], sigma=std_m2)

        param1_m1 = popt_m1-np.sqrt(np.diag(pcov_m1))
        param2_m1 = popt_m1+np.sqrt(np.diag(pcov_m1))

        param1_m2 = popt_m2-np.sqrt(np.diag(pcov_m2))
        param2_m2 = popt_m2+np.sqrt(np.diag(pcov_m2))

        x = np.logspace(2, np.log10(N_max), 100, dtype=np.int64)
        plt.fill_between(x, y1=f(x, *param1_m1), y2=f(x, *param2_m1), color='lightblue', alpha=0.3, label=f'$1\sigma$ of {func_1[0]}')
        plt.fill_between(x, y1=f(x, *param1_m2), y2=f(x, *param2_m2), color='lightgreen', alpha=0.3, label=f'$1\sigma$ of {func_2[0]}')

        plt.plot(x, f(x, *popt_m1), color='lightblue', label='Fit {0}, a = {1:.2e} $\pm$ {2:.2e}'.format(func_1[0], popt_m1[0], np.sqrt(pcov_m1[0, 0])))
        plt.plot(x, f(x, *popt_m2), color='lightgreen', label='Fit {0}, a = {1:.2e} $\pm$ {2:.2e}'.format(func_2[0], popt_m2[0], np.sqrt(pcov_m2[0, 0])))
    except:
        print('Fitting was not sucsessful!')

    plt.plot(x_N, mean_m1, color='navy', linestyle='None', markersize=5, marker='o', label=func_1[0])
    plt.plot(x_N, mean_m2, color='forestgreen', linestyle='None', markersize=5, marker='o', label=func_2[0])

    plt.xlabel(r'$N$')
    plt.ylabel(f'{func_dist[0]}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(x_N[0], x_N[-1])

    plt.legend()


def two_step_mle(rho: np.array, M0: np.array, N: int, mirror=True, alpha=0.5):
    '''
    Estimates with one intermediate step of POVM realignment.

    :param rho  : true state
    :param M0   : inital POVM set
    :param N    : total number of measurements
    :param alpha: this hyperparameter determines the amount of measurements without realignment
    :return: adaptive estimated state
    '''
    # initial estimate
    N0    = int(N**alpha)
    D0    = simulate.measure(rho, N0, M0)
    rho_0 = mle.iterative(D0, M0)

    # rallignment accordning to initial estimate
    _, phi, theta = general.extract_param(rho_0)
    M1    = general.realign_povm(M0, phi, theta, mirror=mirror)

    # true state
    N1    = int(N-N0)
    D1    = simulate.measure(rho, N1, M1)
    rho_1 = mle.iterative(D1, M1)

    return rho_1


def infidelity_qutip(rho_1: np.array, rho_2: np.array):
    '''
    Calculates the infidelity of two given states according to qutip.

    :param rho_1: density representation of the first state
    :param rho_2: density represnetation of the second state
    :return: infidelity
    '''
    Qrho_1 = qt.Qobj(rho_1)
    Qrho_2 = qt.Qobj(rho_2)

    return 1-qt.fidelity(Qrho_1, Qrho_2)


# Pauli-6 states
state6    = np.empty((6, 2, 2), dtype=np.complex)
state6[0] = 1/np.sqrt(2)*np.array([1, 1])
state6[1] = 1/np.sqrt(2)*np.array([1, -1])
state6[2] = 1/np.sqrt(2)*np.array([1, 1j])
state6[3] = 1/np.sqrt(2)*np.array([1, -1j])
state6[4] = np.array([1, 0])
state6[5] = np.array([0, 1])

# Pauli-4 states
state4    = np.empty((6, 2), dtype=np.complex)
state4[0] = 1/np.sqrt(2)*np.array([1, 1])
state4[1] = 1/np.sqrt(2)*np.array([1, 1j])
state4[2] = np.array([1, 0])
state4[3] = np.array([-0.32505758+0.32505758j, 0.88807383+0.j])
