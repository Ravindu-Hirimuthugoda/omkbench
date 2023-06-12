#!/bin/bash

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math


def read(fname: str) -> np.array:
    A = []
    with open(fname) as f:
        for line in f.readlines():
            A.append(line.split(","))
    return np.array(A)


def plot_h2d_d2h(A: np.array) -> None:
    fig = plt.figure(figsize=(9, 9))

    words = A[:, 0].astype(int)
    h2d = A[:, 1].astype(float)
    d2h = A[:, 2].astype(float)
    plt.plot(words, d2h, label="D2H")
    plt.plot(words, h2d, label="H2D")

    plt.title("H2D and D2H Time", fontsize=23)
    plt.xlabel("# of 8-byte Words", fontsize=18)
    plt.ylabel("Time (s)", fontsize=18)
    plt.xscale("log", base=10)
    plt.yscale("log", base=10)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc=0)
    plt.savefig("h2d_d2h.pdf")
    plt.close()


def plot_d2d(A: np.array) -> None:
    fig = plt.figure(figsize=(9, 9))

    block_size = np.unique(A[:, 1].astype(int))
    for bsize in block_size:
        B = A[A[:, 0].astype(int) == bsize]
        words = B[:, 1].astype(int)
        time = B[:, 2].astype(float)
        plt.plot(words, time, label=f"block-size={bsize}")

    plt.title("D2D Time", fontsize=23)
    plt.xlabel("# of 8-byte Words", fontsize=18)
    plt.ylabel("Time (s)", fontsize=18)
    plt.xscale("log", base=10)
    plt.yscale("log", base=10)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc=0)
    plt.savefig("d2d.pdf")
    plt.close()


def plot_variants(A: np.array, title: str) -> None:
    kernels = np.unique(A[:, 0])

    for kernel in kernels:
        fig = plt.figure(figsize=(9, 9))
        B = A[A[:, 0] == kernel]

        block_size = np.unique(B[:, 2].astype(int))
        for bsize in block_size:
            C = B[B[:, 2].astype(int) == bsize]
            words = C[:, 3].astype(int)
            time = C[:, 4].astype(float)
            plt.plot(words, time, label=f"block-size={bsize}")

        plt.title(f"{title} Time: {kernel}", fontsize=23)
        plt.xlabel("# of 8-byte Words", fontsize=18)
        plt.ylabel("Time (s)", fontsize=18)
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(loc=0)
        plt.savefig(f"{kernel}.pdf")
        plt.close()

def plot_v(A: np.array, title: str) -> None:
    kernels = np.unique(A[:, 0])

    for kernel in kernels:
        fig = plt.figure(figsize=(9, 9))
        B = A[A[:, 0] == kernel]

        block_size = np.unique(B[:, 1])
        for bsize in block_size:
            C = B[B[:, 1] == bsize]
            words = C[:, 2].astype(int)
            time = C[:, 3].astype(float)
            plt.plot(words, time, label=f"backend={bsize}")

        plt.title(f"{title} Time: {kernel}", fontsize=23)
        plt.xlabel("# of 8-byte Words", fontsize=18)
        plt.ylabel("Time (s)", fontsize=18)
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(loc=0)
        plt.savefig(f"{title}.pdf")
        plt.close()

def plot_bnd(A: np.array, title: str) -> None:
    kernels = np.unique(A[:, 0])

    for kernel in kernels:
        fig = plt.figure(figsize=(9, 9))
        B = A[A[:, 0] == kernel]

        block_size = np.unique(B[:, 1])
        for bsize in block_size:
            C = B[B[:, 1] == bsize]
            words = C[:, 3].astype(int)
            time = C[:, 4].astype(float)
            knl_time = C[:, 5].astype(float)
            precentage = (time / knl_time)*100
            plt.plot(words, precentage, label=f"backend={bsize}")

        plt.title(f"{title} Knl_bnd : bnd {kernel}", fontsize=23)
        plt.xlabel("Words", fontsize=18)
        plt.ylabel("Time (s)", fontsize=18)
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(loc=0)
        plt.savefig(f"vec-sclr-bnd.pdf")
        plt.close()

def plot_flop(A: np.array, title: str) -> None:
    kernels = np.unique(A[:, 0])

    # f = open("demofile3.txt", "a")
    for kernel in kernels:
        fig = plt.figure(figsize=(9, 9))
        B = A[A[:, 0] == kernel]
        back_end = np.unique(B[:, 1])
        for bnd in back_end:
            # print(bnd)
            C = B[B[:, 1] == bnd]
            words = C[:, 2].astype(int)
            time = C[:, 3].astype(float)
            # flps = words / time
            # ai = (2 * words * 8)
            flps = (words + np.log2(words)) / time
            ai = flps / (words * 8)
            plt.plot(ai, flps, label=f"backend={bnd}")

        plt.title(f"{title} Time: {kernel}", fontsize=23)
        plt.xlabel("AI(Flops:Byte)", fontsize=18)
        plt.ylabel("Flops", fontsize=18)
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(loc=0)
        plt.savefig(f"try-ai-with-words.pdf")
        plt.close()




if __name__ == "__main__":
    # A = read("data/frontier/omk_h2d_d2h.txt")
    # plot_h2d_d2h(A)

    # B = read("data/frontier/omk_d2d.txt")
    # plot_d2d(B)

    # C = read("data/frontier/omk_daxpy.txt")
    # plot_variants(C, "Daxpy")

    # D = read("data/frontier/omk_reduction.txt")
    # plot_variants(D, "Reduction")

    # E = read("data/frontier/lanczos.txt")
    # plot_flop(E, "Lanczos")

    # F = read("data/frontier/lanczos_vec-norm.txt")
    # plot_flop(F, "vec-norm-roof")

    # G = read("data/frontier/lanczos_vec-sclr-div.txt")
    # plot_flop(G, "vec-sclr-div1")

    # H = read("data/frontier/lanczos_mtx-col-copy.txt")
    # plot_v(H, "mtx-col-copy")

    # I = read("data/frontier/lanczos_calc-w.txt")
    # plot_flop(I, "calc-w")
    

    # flops = (990814+ np.log2(990814)) / 1.205043e-03
    # ai = flops / (990814 * 8)
    # print(flops)
    # print(ai)

    # I = read("data/frontier/lanczos_vec-sclr-div-lsize.txt")
    # plot_variants(I, "vec-sclr-div-lsize")

    J = read("data/frontier/lanczos_vec-sclr-div-bnd.txt")
    plot_bnd(J, "vec_sclr_div_bnd")
