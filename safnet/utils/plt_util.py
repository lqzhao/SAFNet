"""Matplotlib visualization helpers."""
import numpy as np
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb



def imshows(images, titles=None, suptitle=None, filename=None):
    """Show multiple images"""
    fig = plt.figure(figsize=[len(images) * 8, 8])
    for ind, image in enumerate(images):
        ax = fig.add_subplot(1, len(images), ind + 1)
        ax.imshow(image)
        if titles is not None:
            ax.set_title(titles[ind])
        ax.set_axis_off()
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9, wspace=0.01, hspace=0.01)
    if suptitle:
        plt.suptitle(suptitle)
    if filename:
        fig.savefig(filename)
    else:
        plt.show()
    plt.close(fig)

def plot_point(points,image_points=None,knn_image_points=None,geo_simi=None,context_simi=None):
    points = np.array(points.squeeze().permute(1,0).cpu())
    geo_simi = np.array(geo_simi.cpu())
    context_simi = np.array(context_simi.cpu())
    image_points = np.array(image_points.squeeze().reshape(-1,3).cpu())
    # knn_image_points = np.array(knn_image_points.squeeze().reshape(3,-1).permute(1,0).cpu())
    pdb.set_trace()
    np.save("points.npy",points)
    np.save("image_points.npy",image_points)
    # np.save("knn_image_points.npy", knn_image_points)
    np.save("geo_simi.npy",geo_simi)
    np.save("context_simi.npy",context_simi)


    return 0

    fig = plt.figure(dpi=500)
    ax = Axes3D(fig)

    # colors = points[:, 3:] / 255  # RGBA(0-1)
    colors = np.stack((context_simi,context_simi,context_simi),0)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               cmap='spectral',
               c='r',
               s=0.5,
               linewidth=0,
               alpha=1,
               marker=".")

    plt.title('Point Cloud')
    ax.axis('scaled')  # {equal, scaled}
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig("points.png")
    plt.show()
