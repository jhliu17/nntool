import os
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt

from jax.scipy.stats import t, norm, laplace
from nntool.plot import latexify, savefig, is_latexify_enabled, enable_latexify


def test_latexify(tmp_path):
    os.environ["LATEXIFY"] = "1"
    latexify(fig_width=0.45 * 6, fig_height=1.5)

    x = jnp.linspace(-4, 4, 100)
    normal = norm.pdf(x, loc=0, scale=1)
    laplace_ = laplace.pdf(x, loc=0, scale=1 / (2**0.5))
    student_t1 = t.pdf(x, df=1, loc=0, scale=1)
    student_t2 = t.pdf(x, df=2, loc=0, scale=1)
    LEGEND_SIZE = 7 if is_latexify_enabled() else None

    plt.figure()

    ax = plt.gca()
    (t1_plot,) = plt.plot(x, student_t1, "b--", label="Student\n" + r"$(\nu=1)$")
    (t2_plot,) = plt.plot(x, student_t2, "g--", label="Student\n" + r"$(\nu=2)$")
    (norm_plot,) = plt.plot(x, normal, "k:", label="Gaussian")
    (laplace_plot,) = plt.plot(x, laplace_, "r-", label="Laplace")

    legend1 = plt.legend(
        handles=[t1_plot, norm_plot], loc="upper right", prop={"size": LEGEND_SIZE}
    )
    ax.add_artist(legend1)
    legend2 = plt.legend(
        handles=[t2_plot, laplace_plot], loc="upper left", prop={"size": LEGEND_SIZE}
    )
    ax.add_artist(legend2)

    plt.ylabel("pdf")
    plt.xlabel("$x$")
    sns.despine()
    savefig("studentLaplacePdf2.pdf", fig_dir=tmp_path)

    plt.figure()
    plt.plot(
        x,
        jnp.log(normal),
        "k:",
        x,
        jnp.log(student_t1),
        "b--",
        x,
        jnp.log(student_t2),
        "g--",
        x,
        jnp.log(laplace_),
        "r-",
    )

    plt.ylabel("log pdf")
    plt.xlabel("$x$")
    plt.legend(
        ("Gaussian", "Student " + r"$(\nu=1)$", "Student " + r"$(\nu=2)$", "Laplace"),
        prop={"size": LEGEND_SIZE},
    )
    sns.despine()
    savefig("studentLaplaceLogpdf2.pdf", fig_dir=tmp_path)

    fig, ax = plt.subplots(1, 1)
    ax.plot(
        x,
        jnp.log(normal),
        "k:",
        x,
        jnp.log(student_t1),
        "b--",
        x,
        jnp.log(student_t2),
        "g--",
        x,
        jnp.log(laplace_),
        "r-",
    )

    ax.set_ylabel("log pdf")
    ax.set_xlabel("$x$")
    ax.legend(
        ("Gaussian", "Student " + r"$(\nu=1)$", "Student " + r"$(\nu=2)$", "Laplace"),
        prop={"size": LEGEND_SIZE},
    )
    sns.despine()
    savefig("studentLaplaceLogpdf3.pdf", fig_dir=tmp_path)


def test_context(tmp_path):
    with enable_latexify(fig_width=0.45 * 6, fig_height=1.5) as ctx:
        x = jnp.linspace(-4, 4, 100)
        normal = norm.pdf(x, loc=0, scale=1)
        laplace_ = laplace.pdf(x, loc=0, scale=1 / (2**0.5))
        student_t1 = t.pdf(x, df=1, loc=0, scale=1)
        student_t2 = t.pdf(x, df=2, loc=0, scale=1)
        LEGEND_SIZE = ctx.legend_size

        plt.figure()

        ax = plt.gca()
        (t1_plot,) = plt.plot(x, student_t1, "b--", label="Student\n" + r"$(\nu=1)$")
        (t2_plot,) = plt.plot(x, student_t2, "g--", label="Student\n" + r"$(\nu=2)$")
        (norm_plot,) = plt.plot(x, normal, "k:", label="Gaussian")
        (laplace_plot,) = plt.plot(x, laplace_, "r-", label="Laplace")

        legend1 = plt.legend(
            handles=[t1_plot, norm_plot], loc="upper right", prop={"size": LEGEND_SIZE}
        )
        ax.add_artist(legend1)
        legend2 = plt.legend(
            handles=[t2_plot, laplace_plot],
            loc="upper left",
            prop={"size": LEGEND_SIZE},
        )
        ax.add_artist(legend2)

        plt.ylabel("pdf")
        plt.xlabel("$x$")

        ctx.savefig("in_context_studentLaplacePdf2.pdf", fig_dir=tmp_path)

    x = jnp.linspace(-4, 4, 100)
    normal = norm.pdf(x, loc=0, scale=1)
    laplace_ = laplace.pdf(x, loc=0, scale=1 / (2**0.5))
    student_t1 = t.pdf(x, df=1, loc=0, scale=1)
    student_t2 = t.pdf(x, df=2, loc=0, scale=1)
    LEGEND_SIZE = None

    plt.figure()

    ax = plt.gca()
    (t1_plot,) = plt.plot(x, student_t1, "b--", label="Student\n" + r"$(\nu=1)$")
    (t2_plot,) = plt.plot(x, student_t2, "g--", label="Student\n" + r"$(\nu=2)$")
    (norm_plot,) = plt.plot(x, normal, "k:", label="Gaussian")
    (laplace_plot,) = plt.plot(x, laplace_, "r-", label="Laplace")

    legend1 = plt.legend(
        handles=[t1_plot, norm_plot], loc="upper right", prop={"size": LEGEND_SIZE}
    )
    ax.add_artist(legend1)
    legend2 = plt.legend(
        handles=[t2_plot, laplace_plot],
        loc="upper left",
        prop={"size": LEGEND_SIZE},
    )
    ax.add_artist(legend2)

    plt.ylabel("pdf")
    plt.xlabel("$x$")

    ctx.savefig("out_context_studentLaplacePdf2.pdf", fig_dir=tmp_path)
