import general
import mixed
import pure
import inversion
import mle
import onestep as os


def ost():
    name  = 'test_ost'
    path  = ''
    new   = True
    debug = True

    d = {}
    d['dim']        = 2
    d['N_min']      = int(1e01)
    d['N_max']      = int(1e05)
    d['N_ticks']    = 20
    d['N_mean']     = 750
    d['povm_name']  = 'Pauli-4'
    d['f_sample']   = pure.sample_unitary
    d['f_estimate'] = inversion.linear
    d['f_distance'] = general.infidelity

    ost = os.OneStepTomography(name, path, new, debug, d)
    ost.parameter_report()
    ost.update_param('N_min', int(1e01))
    ost.get_originals()
    ost.get_estimates()
    ost.get_distances()
    ost.get_valids()
    ost.get_scaling()
    ost.create_originals()
    ost.reconstruct()
    ost.calculate_fitparam()
    ost.plot_distance()
    ost.plot_validity()
    ost.dispatch_model()
    ost = os.OneStepTomography(name, path, False, debug, d=None)
    ost.plot_distance()
    ost.dispatch_model(path=path)


def osc():
    name  = 'osc'
    path  = 'results/mle/onestep/'
    debug = True

    name_list = ['mle_pauli4', 'mle_pauli6', 'mle_sic']

    osc = os.OneStepComparison(name, path, debug, name_list)
    osc.parameter_report()
    osc.get_estimation_method()
    osc.get_povm_name()
    osc.get_N_min()
    osc.compare_distance(osc.transform_citeria('f_estimate'), osc.transform_citeria('povm_name'))
    osc.dispatch_model()
