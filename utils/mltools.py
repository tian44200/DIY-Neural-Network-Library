import numpy as np

# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import pydot


def plot_data(data, labels=None):
    """
    Affiche des donnees 2D
    :param data: matrice des donnees 2d
    :param labels: vecteur des labels (discrets)
    :return:
    """
    if labels is not None:
        labels = labels.reshape(-1)
    cols, marks = ["red", "green", "blue", "orange", "black", "cyan"], [
        ".",
        "+",
        "*",
        "o",
        "x",
        "^",
    ]
    if labels is None:
        plt.scatter(data[:, 0], data[:, 1], marker="x")
        return
    for i, l in enumerate(sorted(list(set(labels.flatten())))):
        plt.scatter(
            data[labels == l, 0], data[labels == l, 1], c=cols[i], marker=marks[i]
        )


def plot_frontiere(data, f, step=20):
    """Trace un graphe de la frontiere de decision de f
    :param data: donnees
    :param f: fonction de decision
    :param step: pas de la grille
    :return:
    """
    grid, x, y = make_grid(data=data, step=step)
    plt.contourf(
        x,
        y,
        f(grid).reshape(x.shape),
        colors=("lightgray", "skyblue"),
        levels=[-1, 0, 1],
    )


def plot_frontiere_perceptron(data, f, step=1_000):
    mmax = data.max(0)
    mmin = data.min(0)
    x, y = np.meshgrid(
        np.linspace(mmin[0], mmax[0], step), np.linspace(mmin[1], mmax[1], step)
    )
    grid = np.hstack((x.reshape(x.size, 1), y.reshape(y.size, 1)))
    pred = f.predict(grid)
    plt.contourf(
        x,
        y,
        pred.reshape(x.shape),
        colors=("lightgray", "skyblue"),
        levels=[-1000, 0, 1000],
    )


def make_grid(data=None, xmin=-5, xmax=5, ymin=-5, ymax=5, step=20):
    """Cree une grille sous forme de matrice 2d de la liste des points
    :param data: pour calcluler les bornes du graphe
    :param xmin: si pas data, alors bornes du graphe
    :param xmax:
    :param ymin:
    :param ymax:
    :param step: pas de la grille
    :return: une matrice 2d contenant les points de la grille
    """
    if data is not None:
        xmax, xmin, ymax, ymin = (
            np.max(data[:, 0]),
            np.min(data[:, 0]),
            np.max(data[:, 1]),
            np.min(data[:, 1]),
        )
    x, y = np.meshgrid(
        np.arange(xmin, xmax, (xmax - xmin) * 1.0 / step),
        np.arange(ymin, ymax, (ymax - ymin) * 1.0 / step),
    )
    grid = np.c_[x.ravel(), y.ravel()]
    return grid, x, y


def gen_arti(centerx=1, centery=1, sigma=0.1, nbex=1000, data_type=0, epsilon=0.02):
    """Generateur de donnees,
    :param centerx: centre des gaussiennes
    :param centery:
    :param sigma: des gaussiennes
    :param nbex: nombre d'exemples
    :param data_type: 0: melange 2 gaussiennes, 1: melange 4 gaussiennes, 2:echequier
    :param epsilon: bruit dans les donnees
    :return: data matrice 2d des donnnes,y etiquette des donnnees
    """
    if data_type == 0:
        # melange de 2 gaussiennes
        xpos = np.random.multivariate_normal(
            [centerx, centerx], np.diag([sigma, sigma]), nbex // 2
        )
        xneg = np.random.multivariate_normal(
            [-centerx, -centerx], np.diag([sigma, sigma]), nbex // 2
        )
        data = np.vstack((xpos, xneg))
        y = np.hstack((np.ones(nbex // 2), -np.ones(nbex // 2)))
    if data_type == 1:
        # melange de 4 gaussiennes
        xpos = np.vstack(
            (
                np.random.multivariate_normal(
                    [centerx, centerx], np.diag([sigma, sigma]), nbex // 4
                ),
                np.random.multivariate_normal(
                    [-centerx, -centerx], np.diag([sigma, sigma]), nbex // 4
                ),
            )
        )
        xneg = np.vstack(
            (
                np.random.multivariate_normal(
                    [-centerx, centerx], np.diag([sigma, sigma]), nbex // 4
                ),
                np.random.multivariate_normal(
                    [centerx, -centerx], np.diag([sigma, sigma]), nbex // 4
                ),
            )
        )
        data = np.vstack((xpos, xneg))
        y = np.hstack((np.ones(nbex // 2), -np.ones(nbex // 2)))

    if data_type == 2:
        # echiquier
        data = np.reshape(np.random.uniform(-4, 4, 2 * nbex), (nbex, 2))
        y = np.ceil(data[:, 0]) + np.ceil(data[:, 1])
        y = 2 * (y % 2) - 1
    # un peu de bruit
    data[:, 0] += np.random.normal(0, epsilon, nbex)
    data[:, 1] += np.random.normal(0, epsilon, nbex)
    # on mélange les données
    idx = np.random.permutation((range(y.size)))
    data = data[idx, :]
    y = y[idx]
    return data, y.reshape(-1, 1)


def plot_img(X_test,y_test,net,nb_pred=6,n=16):
    random_ind = np.random.choice(np.arange(X_test.shape[0]), nb_pred, replace=False)
    plt.figure(figsize=(20,5*np.ceil(nb_pred / 4)))
    j = 1
    for i in random_ind:
        plt.subplot(int(np.ceil(nb_pred / 4)),3,j)
        plt.title("classe predite : {0} / vraie classe : {1}".format(net.predict(np.asarray([X_test[i]])), y_test[i]))
        show_image(X_test[i],n)
        j+=1

def show_image(data,n=16):
    plt.imshow(data.reshape((n,n)),interpolation="nearest",cmap="gray")

def plot_rec(X_test,net,nbp=None,n=28,n_comp=4):
    
    if  nbp == None:
        #echantillon contenant tous les chiffres 
        ids = [1,3,5,7,2,0,18,15,17,4]
        nbp=10
    else:
        ids = np.random.choice(np.arange(X_test.shape[0]), nbp, replace=False)
    plt.figure(figsize=(5*nbp,15))
    j = 1
    for i in ids:
        plt.subplot(3,nbp,j)
        show_image(X_test[i],n)

        plt.subplot(3,nbp,j+nbp)
        show_image(net.predict(np.asarray([X_test[i]])),n)
        
        j+=1



def plot_usps_predictions(X, indices, originale=True, title=""):
    img_title = "Image reconstruite"
    if originale:
        img_title = "Image originale"

    num_images = len(indices)
    figsize = (15, 3)

    fig, axs = plt.subplots(nrows=1, ncols=num_images, figsize=figsize)
    plt.subplots_adjust(top=1)
    fig.suptitle(title)

    for i, idx in enumerate(indices):
        axs[i].imshow(X[idx].reshape((16, 16)))
        axs[i].set_title(f"{img_title} {idx}", y=-0.2)
        axs[i].axis("off")

    fig.tight_layout()
    plt.show()



def plot_net(
    optim,
    X,
    y,
    net_type="classif",
    net_title="",
    data_xlabel="",
    data_ylabel="",
    display_loss=True,
    display_boundary=True,
    display_score=True,
):
    if net_type == "reglin":
        X = np.column_stack((X, y))
        display_score = False

    elif net_type == "multiclass":
        display_boundary = False

    elif net_type == "auto_encodeur":
        display_boundary = False
        display_score = False

    ncols = np.array([display_loss, display_score, display_boundary]).sum()

    if ncols == 0:
        return

    figsize = (20, 6)
    if ncols == 1:
        figsize = (7, 6)
    elif ncols == 2:
        figsize = (14, 6)

    fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)

    if ncols == 1:
        axs = [axs]

    i = 0

    if display_loss:
        loss_name = optim.loss.__class__.__name__

        axs[i].plot(optim.train_loss, label=f"{loss_name} in Train", c="steelblue")

        if optim.test_loss is not None and len(optim.test_loss) != 0:
            axs[i].plot(optim.test_loss, label=f"{loss_name} in Test", c="coral")

        axs[i].set_xlabel("Nombre d'itérations")
        axs[i].set_ylabel("Loss")
        axs[i].set_title(f"Evolution de la {loss_name}")
        axs[i].legend()

        i += 1

    if display_boundary:
        if net_type == "reglin":
            w = optim.net.modules[0]._parameters["W"][0][0]
            toPlot = [w * x for x in X[:, 0]]

            axs[i].scatter(X[:, 0], X[:, 1], c="midnightblue", label="data")
            axs[i].set_xlabel(data_xlabel)
            axs[i].set_ylabel(data_ylabel)
            axs[i].plot(X[:, 0], toPlot, lw=4, color="r", label="reglin")
            axs[i].set_title(f"Droite de la régression avec â = {w:.2f}")
            axs[i].legend()

        elif net_type == "classif":
            colors = ["darksalmon", "skyblue"]
            markers = ["o", "x"]

            classes = [-1, 1]
            if optim.net.classes_type == "0/1":
                classes = [0, 1]

            axs[i].set_title(f"Frontiere de décision pour {len(classes)} classes")

            y = y.reshape(-1)
            for j, cl in enumerate(classes):
                X_cl = X[y == cl]
                axs[i].scatter(
                    X_cl[:, 0],
                    X_cl[:, 1],
                    c=colors[j],
                    marker=markers[j],
                    label=f"Classe : {cl}",
                )

            mmax = X.max(0)
            mmin = X.min(0)

            step = 1000
            x1grid, x2grid = np.meshgrid(
                np.linspace(mmin[0], mmax[0], step), np.linspace(mmin[1], mmax[1], step)
            )

            grid = np.hstack(
                (x1grid.reshape(x1grid.size, 1), x2grid.reshape(x2grid.size, 1))
            )

            res = optim.net.predict(grid)
            res = res.reshape(x1grid.shape)

            axs[i].contourf(
                x1grid,
                x2grid,
                res,
                colors=colors,
                levels=[-1000, 0, 1000],
                alpha=0.4,
            )

            axs[i].set_xlabel(data_xlabel)
            axs[i].set_ylabel(data_ylabel)
            axs[i].legend()

        i += 1

    if display_score:
        axs[i].plot(optim.train_score, label="score in Train", c="steelblue")

        if optim.test_score is not None and len(optim.test_score) != 0:
            axs[i].plot(optim.test_score, label="score in Test", c="coral")

        axs[i].set_ylim(0, 1.1)
        axs[i].set_xlabel("Nombre d'itérations")
        axs[i].set_ylabel("Score")
        axs[i].set_title("Evolution du score")
        axs[i].legend()

        i += 1

    fig.suptitle(net_title)
    plt.show()


def one_hot_y(y, nb_classes):
    y = y.reshape(-1)
    min_y = np.min(y)
    N = y.shape[0]

    y_shift = y - min_y

    y_oh = np.zeros((N, nb_classes), dtype="int")
    y_oh[np.arange(N), y_shift] = 1

    return y_oh


def normalisation(data):
    dt = data.copy()

    for i in range(data.shape[1]):
        mini = np.min(data[:, i])
        maxi = np.max(data[:, i])
        if maxi == mini:
            if maxi > 0:
                dt[:, i] = 1
            else:
                dt[:, i] = 0
        else:
            dt[:, i] = (data[:, i] - mini) / (maxi - mini)

    return dt.astype("float64")

def net_to_graph(net, net_name="network", horizontal=False):
    net = net.modules

    if horizontal:
        graph = pydot.Dot(graph_type="digraph", rankdir="LR")
    else:
        graph = pydot.Dot(graph_type="digraph")

    for i, layer in enumerate(net):
        label = f"{i} - {layer}"

        if layer.__class__.__name__ in ["Linear", "Conv1D", "MaxPool1D", "Flatten"]:
            node = pydot.Node(label, shape="box")
        else:
            node = pydot.Node(label)

        graph.add_node(node)

    nodes = graph.get_nodes()

    for i in range(len(nodes) - 1):
        src_node = nodes[i]
        dst_node = nodes[i + 1]
        edge = pydot.Edge(src_node, dst_node)
        graph.add_edge(edge)

    graph.write_png(f"{net_name}.png")



def classification_report(y_true, y_pred, target_names):
    n_classes = len(target_names)
    support = [sum(y_true == i) for i in range(n_classes)]
    precision = [
        sum((y_true == i) & (y_pred == i)) / max(sum(y_pred == i), 1)
        for i in range(n_classes)
    ]
    recall = [
        sum((y_true == i) & (y_pred == i)) / max(support[i], 1)
        for i in range(n_classes)
    ]
    f1_score = [
        2 * precision[i] * recall[i] / max((precision[i] + recall[i]), 1e-9)
        for i in range(n_classes)
    ]
    accuracy = sum(y_true == y_pred) / len(y_true)

    report_df = pd.DataFrame(
        {
            "class": target_names + ["accuracy"],
            "precision": precision + [accuracy],
            "recall": recall + [""],
            "f1-score": f1_score + [""],
            "support": support + [len(y_true)],
        }
    )

    report_df.set_index("class", inplace=True)

    cm = confusion_matrix(y_true, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Prédictions")
    ax.set_ylabel("Labels")
    ax.set_title("Matrice de confusion")
    ax.xaxis.set_ticklabels(target_names)
    ax.yaxis.set_ticklabels(target_names)

    plt.show()

    return report_df
