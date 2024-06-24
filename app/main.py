from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from sklearn import datasets
import mpl_toolkits.mplot3d  # noqa: F401
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import io

# Set the matplotlib backend to 'Agg' for non-GUI usage
plt.switch_backend('Agg')

app = FastAPI()

@app.get("/iris")
def get_iris_image():
    iris = datasets.load_iris()

    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

    X_reduced = PCA(n_components=3).fit_transform(iris.data)
    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c=iris.target,
        s=40,
    )

    ax.set_title("First three PCA dimensions")
    ax.set_xlabel("1st Eigenvector")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("2nd Eigenvector")
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel("3rd Eigenvector")
    ax.zaxis.set_ticklabels([])

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return StreamingResponse(buf, media_type='image/png')