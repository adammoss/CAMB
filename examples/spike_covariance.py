import argparse
import os
import sys

import numpy as np
from scipy import integrate
from scipy.interpolate import CubicSpline

camb_path = os.path.realpath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, camb_path)
import camb


def main(args):
    a = np.logspace(-7, 0, 1000)
    H0 = 67.66
    ombh2 = 0.02242
    omch2 = 0.11933
    tau = 0.0561
    mnu = 0.06
    pars = camb.CAMBparams()
    pars.set_accuracy(AccuracyBoost=1)
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0, tau=tau)
    results_lcdm = camb.get_background(pars)
    densities_lcdm = results_lcdm.get_background_densities(a)
    total_density = CubicSpline(a, densities_lcdm["tot"] / a**4)

    num_spikes = 30
    a_spikes = np.logspace(-5, 0, num_spikes)

    def omega(a, ac, amp, beta=6):
        return amp * ((2 * ac**beta) / (a**beta + ac**beta)) ** (6 / beta)

    def f_norm(a1, a2, *args):
        a_norm, sigma = args
        w1 = omega(a1, a_norm, 1) * total_density(a_norm) / (total_density(a1)) / a1
        w2 = omega(a2, a_norm, 1) * total_density(a_norm) / (total_density(a2)) / a2
        return w1 * w2 * 1 / (1 + sigma * (np.log10(a1) - np.log10(a2)) ** 2)

    def covariance_norm(a_norm, sigma, s=2):
        amin_1 = a_norm / s
        amax_1 = min(a_norm * s, 1.0)
        amin_2 = a_norm / s
        amax_2 = min(a_norm * s, 1.0)
        return integrate.dblquad(f_norm, amin_1, amax_1, lambda a1: amin_2, lambda a1: amax_2, args=(a_norm, sigma))[0]

    def f(a1, a2, *args):
        i1, i2, sigma = args
        w1 = omega(a1, a_spikes[i1], 1) * total_density(a_spikes[i1]) / (total_density(a1)) / a1
        w2 = omega(a2, a_spikes[i2], 1) * total_density(a_spikes[i2]) / (total_density(a2)) / a2
        return w1 * w2 * 1 / (1 + sigma * (np.log10(a1) - np.log10(a2)) ** 2)

    def covariance(i, j, sigma, s=2):
        amin_1 = max(a_spikes[i] / s, a_spikes[0])
        amax_1 = min(a_spikes[i] * s, 1.0)
        amin_2 = max(a_spikes[j] / s, a_spikes[0])
        amax_2 = min(a_spikes[j] * s, 1.0)
        return integrate.dblquad(f, amin_1, amax_1, lambda a1: amin_2, lambda a1: amax_2, args=(i, j, sigma))[0]

    def compute_covariance(sigma, s=2, cutoff=0.1):
        a_eq = 1.0 / (1.0 + 3348)
        norm = covariance_norm(a_eq, sigma)
        cov = np.zeros((num_spikes, num_spikes))
        c = 0
        d = None
        for i in range(num_spikes):
            for j in range(i, num_spikes):
                if d is not None and abs(i - j) > d:
                    continue
                cov[i, j] = covariance(i, j, sigma, s=s)
                if i == 0:
                    if j == 0:
                        c = cov[i, j]
                    elif d is None and cov[i, j] < c * cutoff:
                        d = abs(i - j)
                        print(d)
                if i == j:
                    print(i, cov[i, i])
        for i in range(num_spikes):
            for j in range(i):
                cov[i, j] = cov[j, i]
        return cov / norm

    cov = compute_covariance(args.sigma, args.s, args.cutoff)
    filename = f"{args.sigma}_{args.s}_{args.cutoff}_covmat.txt"

    np.savetxt(filename, cov, fmt="%.4e")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=int, default=1000)
    parser.add_argument("--s", type=int, default=10)
    parser.add_argument("--cutoff", type=float, default=0.1)

    args = parser.parse_args()
    main(args)
